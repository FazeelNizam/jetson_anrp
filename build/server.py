import asyncio
import websockets
import json

# Initial Parking State
parking_state = {
    "park_name": "SLT HQ Car Park",
    "capacity": 100,
    "passes": [
        {"name": "A Pass", "available": 20, "limit": 40},
        {"name": "B Pass", "available": 18, "limit": 20},
        {"name": "C Pass", "available": 15, "limit": 20},
        {"name": "D Pass", "available": 8, "limit": 15}
    ],
    "visitors": {"count": 5}
}

# Mock Database of Valid Plates
valid_plates = {
    "CAB-1234": "A Pass",
    "KW-9999": "B Pass",
    "WP-4567": "C Pass"
}

connected_clients = set()

async def broadcast_state():
    if connected_clients:
        message = json.dumps({"type": "update", "data": parking_state})
        await asyncio.gather(*[client.send(message) for client in connected_clients])

async def handle_client(websocket):
    print("Client connected")
    connected_clients.add(websocket)
    try:
        # Send initial state
        await websocket.send(json.dumps({"type": "init", "data": parking_state}))
        
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")
            
            if data['type'] == 'plate_detected':
                plate = data['plate']
                # Simulate processing delay
                await asyncio.sleep(1) 
                
                response = {"type": "verification", "plate": plate, "allowed": False, "message": "Access Denied"}
                
                if plate in valid_plates:
                    pass_type = valid_plates[plate]
                    response["allowed"] = True
                    response["message"] = f"Access Granted: {pass_type}"
                    
                    # Update slots (Mock logic)
                    for p in parking_state['passes']:
                        if p['name'] == pass_type and p['available'] > 0:
                            p['available'] -= 1
                            break
                    
                    # Broadcast new state
                    await broadcast_state()
                
                await websocket.send(json.dumps(response))
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        connected_clients.remove(websocket)

async def main():
    async with websockets.serve(handle_client, "localhost", 8765):
        print("WebSocket Server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
