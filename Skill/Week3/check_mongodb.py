from dotenv import load_dotenv
from usage_logger import UsageLogger

load_dotenv(override=True)

logger = UsageLogger()

print("MongoDB connected successfully!")

logger.close()