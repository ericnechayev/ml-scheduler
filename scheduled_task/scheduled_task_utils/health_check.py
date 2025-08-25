import time
import httpx
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ping_api_health(url, retries=3, delay=2) -> bool:
    """
    Ping the API health route with retries.
    Parameters:
        - url: health check URL
        - retries: number of attempts
        - delay: seconds to wait between retries
    Returns: 
        - True if success, False if all retries fail
    """
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    return True
                else:
                    logger.info(f"Attempt {attempt}: Health check returned {resp.status_code}")
        except Exception as e:
            logger.info(f"Attempt {attempt}: Health check error: {e}")
        
        if attempt < retries:
            time.sleep(delay)  # wait before next try
    return False


