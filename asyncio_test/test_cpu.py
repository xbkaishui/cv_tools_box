import time 
from loguru import logger
import threading
import asyncio

def fibonacci_recursive(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)


def test_fib_no_thread():
    fibonacci_recursive(40)
    fibonacci_recursive(41)
    
def test_fib_thread():
    one_thread = threading.Thread(target=fibonacci_recursive, args=(40,))
    two_thread = threading.Thread(target=fibonacci_recursive, args=(41,))
    
    one_thread.start()
    two_thread.start()
    
    one_thread.join()
    two_thread.join()
    
async def test_fib_async():
    await asyncio.gather(
        asyncio.to_thread(fibonacci_recursive, 40),
        asyncio.to_thread(fibonacci_recursive, 41)
    )
    
if __name__ == "__main__":
    start = time.time()
    # test_fib_no_thread()
    # test_fib_thread()
    result = asyncio.run(test_fib_async())
    end = time.time()
    logger.info(f"Time taken: {end-start}")