import argparse
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def _run_execute(bitfile, driver_dir, result_queue):
    try:
        import numpy as np

        from FaceDetector import load_finn_accelerator

        result_queue.put({"stage": "load_start"})
        start_load = time.time()
        accel = load_finn_accelerator(bitfile, driver_dir)
        load_s = time.time() - start_load

        input_shape = accel.ishape_normal()
        output_shape = accel.oshape_normal()
        packed_input_shape = accel.ishape_packed()
        packed_output_shape = accel.oshape_packed()

        result_queue.put(
            {
                "stage": "load_done",
                "load_s": load_s,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "packed_input_shape": packed_input_shape,
                "packed_output_shape": packed_output_shape,
                "ip_keys": sorted(accel.ip_dict.keys()),
            }
        )

        inp = np.zeros(input_shape, dtype=np.uint8)
        result_queue.put({"stage": "execute_start"})
        start_exec = time.time()
        if os.environ.get("FACE_FRENZY_MANUAL_DMA_SMOKE", "1") == "1":
            out, dma_trace = _manual_execute_with_trace(accel, inp, result_queue)
        else:
            out = accel.execute(inp)
            dma_trace = []
        exec_s = time.time() - start_exec
        out_arr = np.asarray(out)

        result_queue.put(
            {
                "ok": True,
                "load_s": load_s,
                "exec_s": exec_s,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "packed_input_shape": packed_input_shape,
                "packed_output_shape": packed_output_shape,
                "out_shape": out_arr.shape,
                "out_dtype": str(out_arr.dtype),
                "out_min": int(out_arr.min()),
                "out_max": int(out_arr.max()),
                "out_mean": float(out_arr.mean()),
                "dma_trace": dma_trace,
            }
        )
    except Exception:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})


def _dma_status(accel):
    statuses = {}
    for index, dma in enumerate(accel.idma):
        statuses[f"idma{index}"] = int(dma.read(0x00))
    for index, dma in enumerate(accel.odma):
        statuses[f"odma{index}"] = int(dma.read(0x00))
    return statuses


def _emit_trace(result_queue, trace, label, accel):
    status = _dma_status(accel)
    trace.append((label, status))
    result_queue.put({"stage": "dma_status", "label": label, "status": status})
    return status


def _manual_execute_with_trace(accel, inp, result_queue):
    import numpy as np

    trace = []
    _emit_trace(result_queue, trace, "before_pack", accel)

    ibuf_folded = accel.fold_input(inp)
    result_queue.put({"stage": "manual_step", "label": "fold_input_done"})
    start_pack = time.time()
    ibuf_packed = accel.pack_input(ibuf_folded)
    result_queue.put(
        {
            "stage": "manual_step",
            "label": "pack_input_done",
            "packed_shape": ibuf_packed.shape,
            "elapsed_s": time.time() - start_pack,
        }
    )
    accel.copy_input_data_to_device(ibuf_packed)
    _emit_trace(result_queue, trace, "after_copy_input", accel)

    _manual_launch_iodma(accel, result_queue, trace)
    _emit_trace(result_queue, trace, "after_launch", accel)

    deadline = time.time() + float(os.environ.get("FACE_FRENZY_DMA_TIMEOUT_S", "5.0"))
    last_status = None
    while time.time() < deadline:
        status = _dma_status(accel)
        if status != last_status:
            trace.append(("poll", status))
            result_queue.put({"stage": "dma_status", "label": "poll", "status": status})
            last_status = status
        if all(status[f"odma{index}"] & 0x2 != 0 for index in range(len(accel.odma))):
            _emit_trace(result_queue, trace, "complete", accel)
            outputs = []
            for index in range(accel.num_outputs):
                accel.copy_output_data_from_device(accel.obuf_packed[index], ind=index)
                obuf_folded = accel.unpack_output(accel.obuf_packed[index], ind=index)
                outputs.append(accel.unfold_output(obuf_folded, ind=index))
            return outputs[0] if len(outputs) == 1 else outputs, trace
        time.sleep(0.05)

    _emit_trace(result_queue, trace, "timeout", accel)
    raise TimeoutError(f"Manual DMA poll timed out; trace={trace}")


def _manual_launch_iodma(accel, result_queue, trace):
    batch_size = accel.batch_size

    # This mirrors FINNExampleOverlay.execute_on_buffers() for zynq-iodma but
    # emits status after each register step. It avoids hiding a hang inside the
    # generated helper.
    result_queue.put({"stage": "manual_step", "label": "checking_output_dma_idle"})
    for index, dma in enumerate(accel.odma):
        status = int(dma.read(0x00))
        if status & 0x4 == 0:
            raise RuntimeError(f"Output DMA {index} is not idle before launch: status=0x{status:08x}")

    for iwdma, iwbuf, iwdma_name in accel.external_weights:
        result_queue.put({"stage": "manual_step", "label": f"launch_external_weight_{iwdma_name}"})
        iwdma.write(0x10, iwbuf.device_address)
        iwdma.write(0x1C, batch_size)
        iwdma.write(0x00, 1)
        _emit_trace(result_queue, trace, f"after_launch_{iwdma_name}", accel)

    for index, dma in enumerate(accel.odma):
        result_queue.put(
            {
                "stage": "manual_step",
                "label": f"program_odma{index}",
                "addr": int(accel.obuf_packed_device[index].device_address),
                "batch_size": int(batch_size),
            }
        )
        dma.write(0x10, accel.obuf_packed_device[index].device_address)
        dma.write(0x1C, batch_size)
        _emit_trace(result_queue, trace, f"after_program_odma{index}", accel)
        dma.write(0x00, 1)
        _emit_trace(result_queue, trace, f"after_start_odma{index}", accel)

    for index, dma in enumerate(accel.idma):
        result_queue.put(
            {
                "stage": "manual_step",
                "label": f"program_idma{index}",
                "addr": int(accel.ibuf_packed_device[index].device_address),
                "batch_size": int(batch_size),
            }
        )
        dma.write(0x10, accel.ibuf_packed_device[index].device_address)
        dma.write(0x1C, batch_size)
        _emit_trace(result_queue, trace, f"after_program_idma{index}", accel)
        dma.write(0x00, 1)
        _emit_trace(result_queue, trace, f"after_start_idma{index}", accel)


def main():
    parser = argparse.ArgumentParser(description="Run one FINN accelerator execute with a timeout.")
    parser.add_argument("--bitfile", default="fpga/finn-accel.bit")
    parser.add_argument("--driver-dir", default="fpga")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    print("[fpga-smoke] cwd:", os.getcwd())
    print("[fpga-smoke] bitfile:", args.bitfile)
    print("[fpga-smoke] driver_dir:", args.driver_dir)
    print("[fpga-smoke] timeout_s:", args.timeout)

    result_queue = mp.Queue()
    process = mp.Process(
        target=_run_execute,
        args=(args.bitfile, args.driver_dir, result_queue),
    )
    process.start()
    deadline = time.time() + args.timeout
    last_stage = "start"

    while process.is_alive() and time.time() < deadline:
        try:
            while True:
                event = result_queue.get_nowait()
                next_stage = _print_progress_event(event)
                if next_stage is None:
                    result_queue.put(event)
                    break
                last_stage = next_stage
        except queue.Empty:
            pass
        time.sleep(0.1)

    process.join(0)

    if process.is_alive():
        process.terminate()
        process.join(3)
        print("[fpga-smoke] ERROR: FINN execute timed out.")
        print(f"[fpga-smoke] Last completed stage: {last_stage}")
        if last_stage in ("start", "load_start"):
            print("[fpga-smoke] Timeout occurred during overlay/driver construction.")
        elif last_stage == "load_done":
            print("[fpga-smoke] Timeout occurred before execute_start was reported.")
        else:
            print("[fpga-smoke] Overlay loaded, but accel.execute(...) did not return.")
        raise SystemExit(2)

    result = None
    while True:
        try:
            event = result_queue.get_nowait()
        except queue.Empty:
            break

        if "ok" in event:
            result = event
            continue

        next_stage = _print_progress_event(event)
        if next_stage is not None:
            last_stage = next_stage

    if result is None:
        print("[fpga-smoke] ERROR: child process exited without returning a result.")
        print(f"[fpga-smoke] Last completed stage: {last_stage}")
        raise SystemExit(3)

    if not result.get("ok"):
        print("[fpga-smoke] ERROR: FINN execute failed:")
        print(result.get("traceback", "missing traceback"))
        raise SystemExit(1)

    print("[fpga-smoke] OK: FINN execute returned")
    print("[fpga-smoke] overlay_load_s:", f"{result['load_s']:.3f}")
    print("[fpga-smoke] execute_s:", f"{result['exec_s']:.3f}")
    print("[fpga-smoke] input_shape:", result["input_shape"])
    print("[fpga-smoke] output_shape:", result["output_shape"])
    print("[fpga-smoke] packed_input_shape:", result["packed_input_shape"])
    print("[fpga-smoke] packed_output_shape:", result["packed_output_shape"])
    print("[fpga-smoke] out_shape:", result["out_shape"])
    print("[fpga-smoke] out_dtype:", result["out_dtype"])
    print("[fpga-smoke] out_min:", result["out_min"])
    print("[fpga-smoke] out_max:", result["out_max"])
    print("[fpga-smoke] out_mean:", f"{result['out_mean']:.3f}")
    if result.get("dma_trace"):
        print("[fpga-smoke] dma_trace:")
        for label, status in result["dma_trace"]:
            print("[fpga-smoke]  ", label, _format_statuses(status))


def _format_statuses(statuses):
    return " ".join(f"{name}=0x{value:08x}" for name, value in sorted(statuses.items()))


def _print_progress_event(event):
    if event.get("stage") == "load_start":
        print("[fpga-smoke] loading overlay...")
        return "load_start"
    if event.get("stage") == "load_done":
        print("[fpga-smoke] overlay loaded")
        print("[fpga-smoke] overlay_load_s:", f"{event['load_s']:.3f}")
        print("[fpga-smoke] input_shape:", event["input_shape"])
        print("[fpga-smoke] output_shape:", event["output_shape"])
        print("[fpga-smoke] packed_input_shape:", event["packed_input_shape"])
        print("[fpga-smoke] packed_output_shape:", event["packed_output_shape"])
        print("[fpga-smoke] ip_keys:", event["ip_keys"])
        return "load_done"
    if event.get("stage") == "execute_start":
        print("[fpga-smoke] starting one accel.execute(...)")
        if os.environ.get("FACE_FRENZY_MANUAL_DMA_SMOKE", "1") == "1":
            print("[fpga-smoke] using manual DMA launch with register polling")
        return "execute_start"
    if event.get("stage") == "manual_step":
        extra = ""
        if "addr" in event:
            extra = f" addr=0x{event['addr']:x} batch={event['batch_size']}"
        elif "packed_shape" in event:
            extra = f" packed_shape={event['packed_shape']}"
            if "elapsed_s" in event:
                extra += f" elapsed_s={event['elapsed_s']:.3f}"
        print(f"[fpga-smoke] {event['label']}{extra}")
        return event["label"]
    if event.get("stage") == "dma_status":
        print("[fpga-smoke]  ", event["label"], _format_statuses(event["status"]))
        return event["label"]
    return None


if __name__ == "__main__":
    main()
