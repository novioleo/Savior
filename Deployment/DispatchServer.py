import uvicorn
from fastapi import FastAPI

from Deployment.server_config import DEPLOY_VERSION, SERVER_NAME
from Deployment.DispatchInterfaces.DummyInterface import router as dummy_interface_router

app = FastAPI(title=SERVER_NAME, version=DEPLOY_VERSION)

available_interfaces = [
    (dummy_interface_router,'/dummy_interface'),
]
for m_router, m_path_prefix in available_interfaces:
    app.include_router(m_router, prefix=m_path_prefix)

if __name__ == '__main__':
    from Deployment.server_config import DISPATCH_SERVER_PORT

    uvicorn.run(app, host="0.0.0.0", port=DISPATCH_SERVER_PORT)
