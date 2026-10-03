# Memory Systems implementation

Code trong thư mục này đã triển khai phần bắt buộc của bài lab và chế độ live tùy chọn.
`src/` giữ đúng 7 file Python gốc. Helper trả lời offline và runtime live nằm trong
`agent_baseline.py`, được Advanced dùng lại; thư viện live vẫn chỉ được import khi bật live.

| File | Trách nhiệm |
| --- | --- |
| `config.py` | Đọc `.env`/environment, đường dẫn và tham số compact; mặc định offline |
| `model_provider.py` | Chuẩn hóa và khởi tạo model cho 6 provider; import SDK theo provider |
| `memory_store.py` | Token estimator, hồ sơ Markdown, trích/cập nhật fact, summary và compact theo thread |
| `agent_baseline.py` | Baseline giữ lịch sử trong thread; chứa helper trả lời offline và runtime live dùng chung |
| `agent_advanced.py` | Hồ sơ bền vững và summary + message gần nhất |
| `benchmark.py` | Hai benchmark, recall ở thread mới, lưu bảng và JSON |
| `test_agents.py` | Kiểm chứng memory, đính chính/nhiễu, benchmark và live giả lập |

Chạy từ root repository:

```bash
source .venv/bin/activate
python -m pytest src/test_agents.py -v
python src/benchmark.py
```

`reply(user_id, thread_id, message)` trả dict gồm `answer`, `agent_tokens`, `prompt_tokens`, `mode`.
Hai chỉ số token trong dict là phần tăng của lượt hiện tại; các phương thức `token_usage()` và
`prompt_token_usage()` trả tổng tích lũy theo thread. Một thread thuộc về một user; dùng ID thread
mới nếu chuyển người dùng.

Advanced lưu hồ sơ tại `state/profiles/<user>/User.md` khi dùng cấu hình mặc định. Benchmark tạo
thư mục `state/benchmarks/run-*/` mới cho mỗi lần chạy, gồm `results.json`, `results.md` và hai
thư mục trạng thái `standard/`, `stress/`. Hồ sơ được giữ xuyên conversation trong cùng suite,
không dùng lại hồ sơ của lần chạy trước.

Facts gồm tên, nơi ở, nghề, style, mối quan tâm, đồ uống/món ăn yêu thích và thú cưng. Fact mới thay
fact cũ theo field. Style chỉ thay chiều preference được nói rõ: ví dụ đổi loại ví dụ không làm
mất yêu cầu ngắn gọn. Những quy tắc này là heuristic tiếng Việt có phạm vi hữu hạn.

Summary giữ facts có cấu trúc, các mốc chủ đề (tên viết tắt/mã dự án/tên riêng ghép) và một số
đoạn đầu/gần nhất; kích thước có giới hạn. Sau compact,
`compact_keep_messages` đếm message, không đếm cặp hỏi–đáp. Ngưỡng compact là điều kiện kích hoạt,
không phải giới hạn cứng: một message gần nhất rất dài vẫn được giữ nguyên. Summary thuộc về
thread và nằm trong RAM; chỉ `User.md` bền vững qua khởi động lại. Không tự lưu tin tức tạm thời
vào hồ sơ dài hạn.

## Chế độ live

Cài các package được liệt kê trong README ở root, tạo `.env`, đặt provider/model/key tương ứng,
rồi chạy:

```bash
python src/benchmark.py --live
```

Lệnh này bật live cho cả hai agent. Khi gọi agent trực tiếp, đặt `LAB_OFFLINE=false`; hoặc truyền
`force_offline=True` để luôn dùng offline. Thiếu key/package hoặc lỗi API sẽ báo lỗi rõ ràng,
không âm thầm chuyển sang offline. `custom` cần `CUSTOM_BASE_URL`; có thể bỏ key nếu server
OpenAI-compatible nội bộ không yêu cầu xác thực. Ollama cần server/model đã có sẵn.

Live dùng `create_agent` và `InMemorySaver`; Advanced có prompt hồ sơ động, tool read/write/edit
và middleware nén heuristic. Đây là một chiến lược nén tương đương được phép trong Rubric,
không gọi thêm LLM để tạo summary. Cách nối middleware/runtime dựa trên
[tài liệu LangChain](https://github.com/langchain-ai/docs/blob/main/src/oss/langchain/short-term-memory.mdx).

Test live dùng model giả lập, không xác nhận chất lượng model hay kết nối tài khoản provider.
Xem `RESULTS.md` ở root để đọc output live đã chạy qua API; `ANALYSIS.md` chứa phân tích theo
bốn mốc Rubric, đối chiếu offline/live, ba câu trả lời về bonus và các giới hạn khi kiểm tra hard-code.
