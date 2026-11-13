"""Modal CLI using Windows' working system DNS resolver for artifact uploads."""
import aiohttp.connector
from aiohttp.resolver import ThreadedResolver
aiohttp.connector.DefaultResolver=ThreadedResolver
from modal.cli.entry_point import entrypoint_cli
if __name__=='__main__':entrypoint_cli()
