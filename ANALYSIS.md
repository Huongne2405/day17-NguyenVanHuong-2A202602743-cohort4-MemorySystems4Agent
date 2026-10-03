# Phân tích và đối chiếu bài làm theo Rubric

Bài làm: Day 17 — Memory Systems for AI Agent. Đối chiếu theo thứ tự bốn mốc trong
[Rubric.md](Rubric.md). Output hai benchmark **live** được lưu riêng tại [RESULTS.md](RESULTS.md),
lấy từ lượt người làm bài đã chạy qua API: `state/benchmarks/run-ox7w24_k/`.
Các số liệu offline trước đó được giữ dưới đây để phân biệt hai chế độ. Lượt cập nhật tài liệu
này không gọi thêm API.

**Tên file bài nộp:** `ANALYSIS.md` chứa phân tích; `RESULTS.md` chứa output benchmark.
Nguồn không quy định tên cụ thể, nên đây là tên mô tả theo phương án dự phòng của yêu cầu.
**TODO:** xác nhận với giảng viên/lab coach xem có quy định riêng về tên file; nếu có, đổi tên và
cập nhật liên kết trong bài nộp. Chưa có xác nhận trực tiếp từ giảng viên/lab coach.

Đánh giá dưới đây ghi nhận hiện vật và kết quả kiểm chứng, không tự quy đổi thành điểm được chấm.
Lần kiểm tra gần nhất có 53 test pass; chúng do người triển khai viết và không thay thế đánh giá
độc lập. Bộ test live vẫn dùng model giả lập; kết quả API thật nằm ở lượt benchmark live nêu trên.

## 1. Mốc 0–60: triển khai cơ bản

| Yêu cầu | Hiện vật và kiểm chứng | Trạng thái |
| --- | --- | --- |
| Baseline chỉ nhớ trong cùng thread | `BaselineAgent` trong [agent_baseline.py](src/agent_baseline.py) giữ lịch sử theo `thread_id`; không đọc/ghi hồ sơ. Test cross-session xác nhận thread mới không biết tên đã nói ở thread cũ. | Có |
| Advanced có `User.md` bền vững | [agent_advanced.py](src/agent_advanced.py) dùng `UserProfileStore`, lưu UTF-8 tại thư mục hồ sơ theo user. Test xác nhận tạo lại instance agent vẫn đọc được tên. | Có |
| Compact hoặc cơ chế nén tương đương | `CompactMemoryManager` trong [memory_store.py](src/memory_store.py) nén phần cũ thành summary, giữ message gần nhất, đếm compactions. Stress hiện có 5 lần compact. | Có |
| Benchmark tiếng Việt | [conversations.json](data/conversations.json) và [advanced_long_context.json](data/advanced_long_context.json). | Có |
| Cấu trúc và tài liệu rõ ràng | [README.md](README.md), [Guide.md](Guide.md), [src/README.md](src/README.md); tách cấu hình, provider, memory, agent, runtime, benchmark và test. | Có |

Ba lớp memory có trách nhiệm riêng: short-term là message trong thread; persistent là hồ sơ
`User.md` theo user; compact là summary của thread cộng message gần nhất. Chỉ hồ sơ bền vững qua
khởi động lại. Summary không tự được mang sang thread mới.

Các lỗi trừ điểm mạnh đã có kiểm tra tương ứng: test offline xác nhận Baseline quên qua thread mới;
Advanced ghi/đọc được hồ sơ; compact thật sự kích hoạt; bảng đo được sự khác biệt về recall và
prompt load. Recall live của Baseline có thể khác 0 theo phép chấm từ khóa; chỉ số này không tự
chứng minh Baseline có persistent memory.

## 2. Mốc 60–75: benchmark và test cốt lõi

[benchmark.py](src/benchmark.py) tải mỗi dataset một lần cho một suite và đưa cùng danh sách
conversation cho cả hai agent. Agent được khởi tạo riêng; hồ sơ Advanced giữ xuyên conversation
của cùng suite. Mỗi lần chạy có thư mục trạng thái mới để tránh dữ liệu cũ làm tăng recall.

Sau các lượt chat của từng conversation, mỗi câu recall được hỏi ở một thread mới. Việc chấm diễn
ra ngay tại thời điểm đó, trước conversation tiếp theo. `expected_contains` chỉ được dùng trong
bộ chấm, không truyền vào `reply()`.

| Test bắt buộc | Hiện vật trong [test_agents.py](src/test_agents.py) | Kết quả |
| --- | --- | --- |
| `User.md` read/write/edit | `test_user_markdown_read_write_edit` | Pass |
| Compact trigger | `test_compact_trigger` | Pass |
| Cross-session recall | `test_cross_session_recall` | Pass |

Có thêm kiểm tra user isolation, restart, đính chính, nhiễu, compact lặp lại, token accounting và
prompt load trên thread dài. Toàn bộ 53 trường hợp được pytest chạy hiện pass; môi trường Python
3.14 phát một cảnh báo tương thích Pydantic v1 từ dependency LangChain.

Cả hai bảng trong `RESULTS.md` có đủ sáu chỉ số bắt buộc, bên cạnh cột tên agent:
`Agent tokens only`, `Prompt tokens processed`, `Cross-session recall`, `Response quality`,
`Memory growth (bytes)`, `Compactions`.

## 3. Mốc 75–90: phân tích tác động của compact

### Hai bộ benchmark và độ dài stress

Standard có 10 conversation, 101 lượt người dùng và 14 câu recall. Stress có 1 conversation,
16 lượt dài và 3 câu recall. Văn bản người dùng trong Stress dài 9.261 ký tự, tương đương 2.322
token theo estimator khi tính một lần. Ở lượt live, Baseline xử lý 50.347 prompt token ước lượng
qua toàn bộ chat và recall: lịch sử người dùng cùng câu trả lời của model được mang lại nhiều lần
khiến tổng tải ngữ cảnh tăng. Lượt offline trước đó ghi nhận 23.734 prompt token ở Stress.

### Kết quả live và đối chiếu offline

Các bảng đầy đủ sáu chỉ số của lượt live nằm trong `RESULTS.md`. Các lượt offline trước đó có
trạng thái riêng; token đầu ra và các lượt gọi model/tool của live khác offline, nên không diễn
giải chênh lệch giữa hai chế độ thành hiệu quả chỉ do compact.

| Chế độ / suite | Recall Baseline → Advanced | Prompt Baseline → Advanced | Compactions của Advanced |
| --- | --- | --- | --- |
| Live / Standard | 7,1% → 92,9% | 76343 → 116080 | 4 |
| Live / Stress | 0% → 83,3% | 50347 → 36128 | 5 |
| Offline / Standard trước đó | 0% → 100% | 17209 → 23841 | 0 |
| Offline / Stress trước đó | 0% → 100% | 23734 → 12502 | 5 |

Lượt offline trước đó dùng ngưỡng 1.000 token, giữ 4 message. Artifact live chưa lưu provider/model
hoặc tham số compact, nên không khẳng định toàn bộ cấu hình hai lượt giống nhau.

### Vì sao Advanced có recall tốt hơn

Ở live, recall tăng từ 7,1% lên 92,9% trong Standard và từ 0% lên 83,3% trong Stress. Baseline chỉ
đọc lịch sử thread hiện tại; câu recall được hỏi ở thread mới. Advanced đọc hồ sơ theo `user_id`,
nên có nguồn facts ổn định từ các phiên trước. Kết quả cho thấy lợi ích của persistent memory
trong lượt chạy này, nhưng Advanced chưa khớp đủ facts ở mọi câu hỏi.

Hai agent offline dùng cùng quy tắc trả lời và đạt 0%/100% ở cả hai suite trước đó. Live để model
thật sinh câu trả lời và có thể gọi tool hồ sơ. Chưa có transcript từng câu recall trong artifact
tổng hợp để xác định phần thiếu điểm do quên fact, cập nhật hồ sơ hay diễn đạt khác từ khóa.
Điểm khác 0 của Baseline cũng có thể đến từ gợi ý trong câu hỏi hoặc suy đoán; chưa kết luận
nguyên nhân cụ thể chỉ từ bảng. Recall cao không chứng minh hiểu mọi cách diễn đạt.

### Vì sao compact không luôn thắng ở hội thoại ngắn

Trong Standard live, Advanced xử lý 116.080 prompt token so với 76.343 của Baseline: tăng khoảng
52,1%, dù compact đã xảy ra 4 lần. Advanced mang thêm hồ sơ, prompt memory và schema tool; các
lượt gọi tool làm phát sinh thêm ngữ cảnh và model call. Tổng overhead vẫn lớn hơn phần được nén
trong lượt chạy này. Chưa đo riêng từng nguồn overhead nên không gán toàn bộ 52,1% cho `User.md`.

Ở Standard offline trước đó, prompt của Advanced tăng khoảng 38,5% (23.841 so với 17.209), với
0 lần compact. Thêm persistent memory cải thiện recall nhưng tạo overhead ngay cả khi chưa gọi
tool hay nén lịch sử. Không nên ép ngưỡng thấp chỉ để mọi bảng đều cho thấy Advanced tiết kiệm.

### Vì sao compact chủ yếu tối ưu Prompt tokens processed

Trong Stress live, Advanced xử lý 36.128 prompt token so với 50.347 của Baseline: giảm khoảng
28,2%, với 5 lần compact. Lịch sử cũ được thay bằng summary có giới hạn và message gần nhất.
Đây là phần ngữ cảnh được xử lý lại ở các lượt sau. So sánh live đồng thời bao gồm sự khác biệt
về hồ sơ, tool và câu trả lời, nên chưa tách riêng được tác động nhân quả của compact.

Token đầu ra live được đo riêng: Standard giảm từ 15.241 xuống 11.565 (24,1%); Stress giảm từ
4.313 xuống 4.285 (0,6%). Trong Stress, tỷ lệ giảm đầu ra nhỏ hơn nhiều so với prompt. Điều này
phù hợp với mục tiêu compact là giảm ngữ cảnh mang vào model, không trực tiếp ép model sinh
câu trả lời ngắn hơn. Không suy tỷ lệ này thành chi phí API: tokenizer, đơn giá và định dạng
request thật chưa được đo.

Ở Stress offline trước đó, prompt giảm 47,3% (12.502 so với 23.734), còn đầu ra Advanced tăng
từ 469 lên 509. Vì thế giảm prompt load không đồng nghĩa mọi chỉ số token luôn giảm.

Để tách ảnh hưởng của compact khỏi việc thêm hồ sơ, phép đối chiếu **offline trước đó** dùng cùng
Advanced/cùng Stress với trạng thái sạch; chỉ đổi ngưỡng compact từ 1.000 lên 1.000.000.000 để
không kích hoạt nén:

| Cấu hình Advanced | Prompt tokens processed | Agent tokens only | Recall | Compactions |
| --- | --- | --- | --- | --- |
| Compact: ngưỡng 1.000, giữ 4 message | 12502 | 509 | 100% | 5 |
| Không kích hoạt compact: ngưỡng 1.000.000.000, giữ 4 message | 24452 | 509 | 100% | 0 |

Trong phép đối chiếu này, compact giảm 48,9% prompt load trong khi recall và token đầu ra giữ
nguyên. Đây là bằng chứng offline; chưa chạy phép đối chiếu bật/tắt compact tương đương qua API.

### Memory growth và rủi ro

Hồ sơ live tăng 756 byte ở Standard và 999 byte ở Stress; các số tương ứng ở offline trước đó là
272 và 205 byte. Chỉ số đo chênh lệch kích thước `User.md`, không đo summary trong RAM hoặc file
output. Live cho model quyền gọi tool write/edit hồ sơ, nên nội dung/định dạng có thể khác bản
offline chỉ upsert bằng rule; chưa kiểm chứng nội dung để xác định vì sao hồ sơ live lớn hơn.

Test xác nhận upsert theo field không sinh bản sao khi nhắc cùng fact. Điều đó không bảo đảm
tool ghi toàn bộ hồ sơ ở live luôn tuân thủ cùng định dạng hay giữ kích thước ổn định. Các mối
quan tâm hợp nhất có thể giữ thông tin hết phù hợp; tăng field hoặc lưu mọi sự kiện vẫn làm hồ
sơ phình ra. Summary có thể mất chi tiết và trích sai fact có thể gây nhớ sai qua nhiều phiên.

## 4. Mốc 90–100: bonus và ba câu trả lời bắt buộc

**Bonus được đối chiếu: conflict handling bằng facts có field rõ ràng và cập nhật theo field.**
Không tuyên bố đã có confidence threshold định lượng hoặc memory decay; hai phần đó chưa làm.

### Bonus giải quyết vấn đề gì?

Người dùng có thể đổi nghề/nơi ở, nhắc lại fact cũ hoặc đưa thông tin gây nhiễu. Nếu nối mọi câu
vào hồ sơ, cả nghề cũ và nghề mới sẽ tồn tại, còn model có thể chọn nhầm. `extract_profile_updates`
trích khai báo được hỗ trợ; `update_facts`/`upsert_fact` cập nhật một giá trị hiện tại theo field.
Upsert còn loại bản trùng của cùng field trong hồ sơ đã có. Các mẫu câu hỏi, giả định và câu đùa
được hỗ trợ không ghi đè hồ sơ.

### Bonus cải thiện recall hoặc token cost như thế nào?

`test_correction_survives_noise_and_restart` xác nhận câu trả lời sau restart dùng nghề/nơi ở mới,
không chứa nghề/nơi ở cũ hoặc thông tin gây nhiễu. `test_duplicate_field_correction` xác nhận
field trùng được thu về một giá trị hiện tại. Việc tránh bản sao giúp kiểm soát kích thước hồ sơ,
qua đó tránh phải mang cùng fact lặp lại trong prompt; test upsert xác nhận file size ổn định
khi cập nhật lặp cùng dữ liệu.

Đây là chứng cứ về hành vi cập nhật và kiểm soát tăng trưởng. Chưa có phép benchmark riêng bật/tắt
bonus để định lượng mức cải thiện recall/token do bonus; không quy toàn bộ mức tăng recall live,
recall 100% offline hay mức giảm prompt ở Stress cho conflict handling. Phép đối chiếu tách riêng
compact ở trên chỉ được thực hiện trong offline.

### Bonus tạo thêm rủi ro gì?

Regex có thể hiểu sai một khai báo và ghi đè fact đúng bằng fact sai. Hồ sơ chỉ giữ giá trị hiện
tại, không có lịch sử nguồn để phục hồi. Danh sách nghề/sở thích hữu hạn có thể bỏ sót dữ liệu
mới; hợp nhất interests có thể giữ sở thích cũ dù người dùng đã đổi ý. Các rule bỏ qua câu hỏi
và nhiễu không phủ hết cách diễn đạt tự nhiên. Vì vậy conflict handling hiện tại cần test ngoài
mẫu, theo dõi nguồn fact và cơ chế sửa/khôi phục nếu dùng thực tế.

## Giới hạn kiểm chứng và kết quả kiểm tra hard-code

Các hạn chế sau vẫn còn trong code; lượt này chỉ cập nhật tài liệu theo output live, chưa sửa chúng:

- Helper `respond_from_facts`/`_format_fragments` trong `agent_baseline.py` chỉ định dạng riêng
  **3 bullet** trong offline. Kiểm tra offline yêu cầu **5 bullet** cho thấy agent lưu được
  preference nhưng trả về **0 bullet**. Đây là rule bám vào mẫu benchmark; không suy lỗi định dạng
  này cho model live nếu chưa kiểm tra riêng.
- `memory_store.py` dùng danh sách cố định cho nghề và interests. “Mình thích TypeScript,
  Kubernetes và PostgreSQL” không được trích; “Mình là kế toán” cũng không được nhận diện.
- Trong các phép kiểm tra offline đổi tên, ID, nơi ở, nghề, đồ ăn và thú cưng, Advanced vẫn đạt
  100% recall. Khi đổi Python/AI sang TypeScript/Kubernetes, recall Standard giảm xuống khoảng
  92,9%. Đây là phép kiểm tra khác với Standard live có cùng mức điểm làm tròn 92,9%.
- Đổi riêng `expected_contains` không đổi câu trả lời; điểm thay từ 100% xuống 0% khi mục tiêu
  không khớp. Đổi nhãn Advanced/Baseline không đổi kết quả. Agent bị ép trả lời sai được 0%.
  Các kiểm tra này chưa cho thấy cài sẵn đáp án cá nhân hoặc ép điểm theo tên agent.
- Recall/quality vẫn chấm bằng substring. “Mình không phải Linh và không thích trà đào” được
  chấm 100% với mục tiêu `Linh`, `trà đào`. Quality cao không xác nhận đúng nghĩa, không mâu thuẫn
  hoặc chất lượng suy luận. Các test correction kiểm tra riêng việc loại fact cũ sai.
- Token là `ceil(len(text.strip()) / 4)`, không phải tokenizer thật. Summary giữ một số facts,
  mốc chủ đề và đoạn trích, chưa chứng minh giữ đủ diễn giải/con số/quan hệ trong tin tức.
- Bộ test live dùng model giả lập, còn benchmark `run-ox7w24_k` đã do người làm bài chạy qua API
  thật. Lượt live xác nhận đường chạy của provider đã chọn; chưa chứng minh cả 6 provider đều
  kết nối được. Provider/model cụ thể chưa được lưu trong artifact tổng hợp. Chưa đánh giá chất
  lượng bằng LLM judge hoặc người chấm độc lập; cấu hình judge đã có nhưng chưa dùng để chấm.

Bài có các hiện vật cho ba mốc đầu và một bonus để reviewer xem xét ở mốc 90–100. Các điểm yếu
về rule theo mẫu, phép chấm và generalization vẫn cần được tính vào đánh giá; không tự khẳng định
bài đạt 90 hay 100 điểm chỉ vì test hiện tại pass.
