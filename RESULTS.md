# Benchmark results

Mode: live

Tokens are character estimates; quality is an offline heuristic, not an LLM judge.

## Standard Benchmark

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| -------- | ----------------- | ----------------------- | -------------------- | ---------------- | --------------------- | ----------- |
| Baseline | 15241             | 76343                   | 7.1%                 | 4.2%             | 0                     | 0           |
| Advanced | 11565             | 116080                  | 92.9%                | 96.8%            | 756                   | 4           |

## Long-Context Stress Benchmark

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| -------- | ----------------- | ----------------------- | -------------------- | ---------------- | --------------------- | ----------- |
| Baseline | 4313              | 50347                   | 0.0%                 | 0.0%             | 0                     | 0           |
| Advanced | 4285              | 36128                   | 83.3%                | 91.7%            | 999                   | 5           |

## Nguồn kết quả và cách chạy

Đây là kết quả live đã chạy qua API, lấy từ
`state/benchmarks/run-ox7w24_k/results.json` và `results.md`. Các bảng trên giữ nguyên số liệu
của lượt chạy này. Lượt cập nhật tài liệu không gọi thêm API.

```bash
source .venv/bin/activate
python src/benchmark.py --live
```

Provider, model và key được cấu hình trong `.env` tại thời điểm chạy. Artifact kết quả chưa ghi
provider/model hoặc ngưỡng compact của lượt live, nên không suy những giá trị đó từ bảng số liệu.
Chạy lại có thể cho kết quả khác. Thư mục `state/` bị gitignore; bảng này được giữ cùng bài nộp.

Token đều là ước lượng ký tự. Quality là heuristic từ khóa/độ dài áp dụng cả ở live,
không phải đánh giá bằng LLM judge. Phân tích theo bốn mốc Rubric và đối chiếu offline/live
nằm riêng trong [ANALYSIS.md](ANALYSIS.md).
