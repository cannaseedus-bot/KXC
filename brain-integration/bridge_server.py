"""
Asyncio WebSocket bridge server — calls router.py for actual inference.
"""

import asyncio
import json
import random
import signal
import sys
import os
from typing import Dict

# Add parent directory to sys.path to find router.py
HERE = os.path.dirname(os.path.abspath(__file__))
HIVE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if HIVE_ROOT not in sys.path:
    sys.path.insert(0, HIVE_ROOT)

try:
    import router
except ImportError:
    router = None

try:
    import websockets
    from websockets import WebSocketServerProtocol
except Exception:
    websockets = None  # graceful fallback if not installed

async def handle_connection(ws: WebSocketServerProtocol, path: str = None):
    print(f"Client connected: {ws.remote_address}")
    # track currently running tasks by id
    tasks: Dict[str, asyncio.Task] = {}

    async def stream_response(req_id: str, prompt: str):
        # Using router.route to get actual AI response
        # Note: router.py currently doesn't support streaming, so we'll wait for full result and simulate streaming tokens
        try:
            if not router:
                await ws.send(json.dumps({"type": "error", "id": req_id, "message": "router.py not found"}))
                return

            loop = asyncio.get_event_loop()
            # Run the blocking router call in a thread
            route_type = router.detect_route(prompt)
            result = await loop.run_in_executor(None, lambda: router.route(prompt, route_type=route_type))
            
            if result.get("live"):
                reply = result.get("reply", "")
                
                # Check for code blocks for newContent
                new_content = None
                if "```" in reply:
                    import re
                    blocks = re.findall(r'```(?:\w+)?\n([\s\S]*?)```', reply)
                    if blocks:
                        new_content = blocks[0]

                # Stream tokens
                words = reply.split()
                for w in words:
                    # check cancellation
                    if req_id in tasks and tasks[req_id].cancelled():
                        return
                    chunk = {"type": "chunk", "id": req_id, "token": w + " "}
                    await ws.send(json.dumps(chunk))
                    await asyncio.sleep(0.01) # fast stream
                
                # Final message
                final = {
                    "type": "done", 
                    "id": req_id, 
                    "result": "success", 
                    "text": reply, 
                    "newContent": new_content,
                    "backend": result.get("backend"),
                    "model": result.get("model")
                }
                await ws.send(json.dumps(final))
            else:
                await ws.send(json.dumps({
                    "type": "error", 
                    "id": req_id, 
                    "message": result.get("error", "no backends available")
                }))
        except Exception as e:
            try:
                await ws.send(json.dumps({"type": "error", "id": req_id, "message": str(e)}))
            except Exception:
                pass
        finally:
            tasks.pop(req_id, None)

    try:
        async for message in ws:
            try:
                msg = json.loads(message)
            except Exception:
                # ignore non-json
                continue

            t = msg.get("type")
            req_id = msg.get("id")

            if t == "ping":
                # heartbeat ping
                await ws.send(json.dumps({"type": "pong"}))
                continue

            if t == "request" and req_id:
                payload = msg.get("payload") or {}
                prompt = payload.get("prompt") or payload.get("text") or ""
                # start a streaming task
                task = asyncio.create_task(stream_response(req_id, prompt))
                tasks[req_id] = task
                continue

            if t == "cancel" and req_id:
                task = tasks.get(req_id)
                if task and not task.done():
                    task.cancel()
                else:
                    await ws.send(json.dumps({"type": "cancelled", "id": req_id}))
                continue
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        # cancel any running tasks
        for tid, t in tasks.items():
            if not t.done():
                t.cancel()

async def main():
    if websockets is None:
        print("websockets package not available. Install with: pip install websockets")
        return

    server = await websockets.serve(handle_connection, "localhost", 8765)
    print("Bridge server listening on ws://localhost:8765")

    # run until cancelled
    stop = asyncio.Future()

    def _on_signal(*_):
        if not stop.done():
            stop.set_result(None)

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _on_signal)
        except NotImplementedError:
            pass

    await stop
    server.close()
    await server.wait_closed()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
