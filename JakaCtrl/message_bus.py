import omni
import carb
import websockets
import asyncio
import random

class MessageBus():

    def __init__(self, uibuilder=None):
        super().__init__()
        self.websocketWS = websockets
        self.websocket_id = random.randint(1000, 9999)
        self.websocket_url= "wss://integrationhubwebsocket.azurewebsites.net/" + str(self.websocket_id)
        self.send_messages = False
        asyncio.ensure_future(self.connect_websocket())
        
    def stop_messages(self):
        self.send_messages = False

    def start_messages(self):
        self.send_messages = True

    async def connect_websocket(self):
        self.websocketWS = await websockets.connect(self.websocket_url)
        await self.websocketWS.send("websocket open")

    async def send_positions_by_websocket(self,mess):
        if self.send_messages:
            try:
                await self.websocketWS.send(mess)

            except websockets.ConnectionClosedError:
                self.open = False
                await self.connect_websocket()


