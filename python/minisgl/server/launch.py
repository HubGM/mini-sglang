from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(
    args: ServerArgs,
    ack_queue: mp.Queue[str],
    health_queue: mp.Queue,
) -> None:
    import torch
    from minisgl.scheduler import Scheduler
    from minisgl.server.health import HealthEventKind, HealthReporter

    reporter = HealthReporter(
        health_queue, source=f"scheduler-{args.tp_info.rank}"
    )
    scheduler = None
    try:
        with torch.inference_mode():
            scheduler = Scheduler(args)
            reporter.emit(HealthEventKind.MODEL_LOADED)
            scheduler.sync_all_ranks()

            if args.tp_info.is_primary():
                ack_queue.put("Scheduler is ready")
                reporter.emit(HealthEventKind.SCHEDULER_READY)
                reporter.heartbeat(0)

            if args.silent_output:
                logging.disable(logging.INFO)

            scheduler.run_forever()
    except KeyboardInterrupt:
        logger = init_logger(__name__)
        if scheduler is not None and scheduler.tp_info.is_primary():
            print()  # for a clean newline after ^C
            logger.info("Scheduler exiting gracefully...")
        if scheduler is not None:
            scheduler.shutdown()
    except BaseException as exc:
        reason = type(exc).__name__
        reporter.fatal(reason)
        if args.tp_info.is_primary():
            ack_queue.put(f"FATAL:{reason}")
        raise
    finally:
        reporter.emit(HealthEventKind.STOPPED)


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess():
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()
        health_queue: mp.Queue = mp.Queue()
        scheduler_processes: list[mp.Process] = []
        tokenizer_processes: list[mp.Process] = []

        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
                scheduler_health_queue=health_queue,
            )
            process = mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue, health_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            )
            process.start()
            scheduler_processes.append(process)

        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        process = mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
                "health_queue": health_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        )
        process.start()
        tokenizer_processes.append(process)
        for i in range(num_tokenizers):
            process = mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                    "health_queue": health_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            )
            process.start()
            tokenizer_processes.append(process)

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            try:
                message = ack_queue.get(timeout=300)
            except queue.Empty as exc:
                raise RuntimeError("Backend startup timed out") from exc
            if message.startswith("FATAL:"):
                raise RuntimeError(f"Backend startup failed: {message}")
            logger.info(message)

        from .health import BackendSupervisor

        supervisor = BackendSupervisor(
            event_queue=health_queue,
            scheduler_processes=scheduler_processes,
            tokenizer_processes=tokenizer_processes,
            heartbeat_timeout_s=server_args.scheduler_heartbeat_timeout_s,
        )
        supervisor.start()
        return supervisor

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
