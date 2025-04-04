import tornado
from tornado_swagger.setup import setup_swagger
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,  # Niveau de log (DEBUG, INFO, WARNING...)
    format="%(message)s",  # Format simplifié, Rich s'occupe du style
    datefmt="[%X]",
    handlers=[RichHandler()]  # 👈 liste contenant le handler Rich
)

logger = logging.getLogger(__name__)

class PredictView(tornado.web.RequestHandler):

    def post(self):
        text = self.get_argument("text")


class TornadoApplication(tornado.web.Application):
    _routes = [
        tornado.web.url(r"/api/predict", PredictView),
    ]
    security_definition = {
    }
    security = [{"TokenQueryAuth": []}]

    def __init__(self):
        settings = {"debug": True}
        setup_swagger(
            self._routes,
            swagger_url="/api/doc",
            api_base_url="/",
            description="Documentation API pour le serveur alerting",
            api_version="1.0.0",
            contact="shift.python.software@gmail.com",
            title="API Tornado Alerting",
            security_definitions=self.security_definition,
        )
        super().__init__(self._routes, **settings)

def start_server(address="0.0.0.0", port=8888):
    """
    Start the tornado server
    :param directory:
    :param address:
    :param port:
    :return:
    """
    app = TornadoApplication()
    app.listen(address=address, port=port)
    logger.info(f"Serving server on {address}:{port}")
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    start_server(address="127.0.0.1", port=8004)