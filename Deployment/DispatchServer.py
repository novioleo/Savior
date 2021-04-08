import uvicorn
from fastapi import FastAPI
import importlib
from Deployment.server_config import DEPLOY_VERSION, SERVER_NAME

app = FastAPI(title=SERVER_NAME, version=DEPLOY_VERSION)

# dummy_interface是用来测试当前服务器是否成功启动
available_interfaces = [
    ('DummyInterface', '/dummy_interface'),
    ('OCRRelatedInterface', '/ocr_interface'),
]
for m_router_package, m_path_prefix in available_interfaces:
    try:
        m_interface_module = importlib.import_module(f'Deployment.DispatchInterfaces.{m_router_package}')
        m_interface_router = getattr(m_interface_module, 'router')
        app.include_router(m_interface_router, prefix=m_path_prefix)
        print(f'{m_router_package} load finish')
    except AttributeError as ae:
        print(f'the APIRouter must name as router in {m_router_package}')
    except Exception as e:
        print(e)
        print(f'{m_router_package} not found.')

if __name__ == '__main__':
    from Deployment.server_config import DISPATCH_SERVER_PORT

    uvicorn.run(app, host="0.0.0.0", port=DISPATCH_SERVER_PORT)
