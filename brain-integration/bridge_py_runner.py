import sys
import os
import json

# Add parent directory to sys.path to find router.py
HERE = os.path.dirname(os.path.abspath(__file__))
HIVE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if HIVE_ROOT not in sys.path:
    sys.path.insert(0, HIVE_ROOT)

try:
    import router
except ImportError:
    # Fallback if import fails
    router = None

def handle(prompt, route_type="general"):
    """
    Handle incoming prompt and return a response object.
    Uses router.py to call actual backends.
    """
    if not router:
        return {"status": "error", "message": "router.py not found"}
    
    try:
        result = router.route(prompt, route_type=route_type)
        if result.get("live"):
            # Extract code blocks for newContent if it looks like code
            reply = result.get("reply", "")
            new_content = None
            if "```" in reply:
                import re
                blocks = re.findall(r'```(?:\w+)?\n([\s\S]*?)```', reply)
                if blocks:
                    new_content = blocks[0] # Take first code block for simplicity
            
            return {
                "status": "ok",
                "response": reply,
                "newContent": new_content,
                "backend": result.get("backend"),
                "model": result.get("model")
            }
        else:
            return {"status": "error", "message": result.get("error", "no reply")}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    data = None
    # If a filepath argument is provided, read from file; otherwise read stdin
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = f.read()
        except Exception as e:
            print(json.dumps({"status":"error","message":f"failed reading input file: {e}"}))
            sys.exit(1)
    else:
        data = sys.stdin.read()

    # Echo a JSON response for compatibility with bridge.ps1
    try:
        # If input is JSON, parse and include id
        payload = json.loads(data)
        if isinstance(payload, dict):
            prompt = payload.get('prompt') or payload.get('text') or data
            route_type = payload.get('route_type') or router.detect_route(prompt) if router else "general"
            resp = handle(prompt, route_type)
            if "id" in payload:
                resp["id"] = payload["id"]
        else:
            resp = handle(data)
    except Exception as e:
        resp = handle(data)
    
    print(json.dumps(resp))
