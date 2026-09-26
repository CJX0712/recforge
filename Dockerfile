FROM python:3.13-slim

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# 默认：端到端演示，生成 benchmark.json
CMD ["python", "examples/run_demo.py"]
