Form báo cáo so sánh hiệu năng theo yêu cầu trong tài liệu cần bao gồm các nội dung và phương pháp đánh giá cụ thể như sau:

*   **Đối tượng so sánh:** Báo cáo cần thực hiện so sánh giữa các loại chỉ mục (index) cụ thể, bao gồm **Flat so với HNSW** và **IVF so với IVF+PQ**.
*   **Chỉ số đo lường:** Việc đánh giá hiệu năng phải được dựa trên hai chỉ số chính là **độ trễ (latency)** và **độ thu hồi (recall@k)**.
*   **Gợi ý thực hiện và trình bày kết quả:** 
    *   Đảm bảo **dùng cùng một tập dữ liệu (dataset)** để kết quả so sánh được khách quan.
    *   Thực hiện đo đạc với các quy mô dữ liệu (N) lớn dần: **10K, 100K và 1M** vector.
    *   Tiến hành **thay đổi các tham số** như `M` và `efSearch` trong quá trình thử nghiệm để xem xét sự biến động.
    *   Kết quả thu được cần được trực quan hóa bằng cách **vẽ biểu đồ (plot)** biểu diễn mối tương quan giữa độ trễ (latency) và độ thu hồi (recall).

Để bạn có thể áp dụng và thực hiện bài báo cáo so sánh hiệu năng một cách hoàn chỉnh, dưới đây là mô tả chi tiết về các thuật toán, ý nghĩa của các chỉ số, cũng như từng bước thực hành cụ thể được tổng hợp từ các tài liệu:

### 1. Bản chất của các thuật toán Index cần so sánh

**A. Flat so với HNSW**
Đây là phép so sánh giữa phương pháp tìm kiếm chính xác tuyệt đối (Exact search) và phương pháp tìm kiếm xấp xỉ dựa trên đồ thị (Approximate Nearest Neighbor - ANN).
*   **Flat (Brute-force):** Là phương pháp quét toàn bộ cơ sở dữ liệu. Nó tính toán độ tương đồng giữa câu truy vấn và *từng vector một* trong cơ sở dữ liệu. 
    *   *Đặc điểm:* Độ phức tạp thời gian là O(n), tức là dữ liệu càng lớn thì càng chậm. Phương pháp này trả về kết quả chính xác 100% nhưng tốc độ rất chậm, thường được dùng làm điểm cơ sở (baseline) để so sánh.
*   **HNSW (Hierarchical Navigable Small Worlds):** Là thuật toán tạo ra một cấu trúc đồ thị phân tầng nhiều lớp, kết hợp hai ý tưởng: Đồ thị thế giới nhỏ (NSW) và Danh sách liên kết bỏ qua (Skip Link List). 
    *   *Cơ chế hoạt động:* Đồ thị có các tầng trên thưa thớt (ít liên kết) và các tầng dưới cùng dày đặc (nhiều liên kết). Khi tìm kiếm, thuật toán bắt đầu ở tầng cao nhất, nhảy vọt qua các khoảng không gian lớn để tìm điểm gần nhất, sau đó đi dần xuống các tầng thấp hơn để tinh chỉnh và tìm ra **Top K** vector tương đồng nhất.
    *   *Đặc điểm:* Giúp giảm độ phức tạp tìm kiếm xuống O(log n), đem lại tốc độ rất nhanh và độ thu hồi (recall) cao, nhưng nhược điểm là tốn nhiều RAM để lưu trữ cấu trúc đồ thị. Thuật toán này rất phù hợp cho môi trường production.

**B. IVF so với IVF + PQ**
Đây là phép so sánh xem việc đánh đổi độ chính xác để tối ưu bộ nhớ diễn ra như thế nào.
*   **IVF (Inverted File Index):** Thay vì tìm kiếm trên toàn bộ dữ liệu, hệ thống sẽ gom cụm (clustering) các vector lại với nhau thành nhiều cụm và xác định các "tâm cụm" (centroid). 
    *   *Cơ chế:* Khi có truy vấn, hệ thống chỉ tìm cụm có tâm gần với truy vấn nhất, sau đó mới tìm kiếm chi tiết bên trong cụm đó. Việc này giúp giảm đáng kể không gian tìm kiếm. Phương pháp này có tốc độ và độ chính xác ở mức khá đến cao, phù hợp với quy mô lớn.
*   **IVF + PQ (Product Quantization):** Là sự kết hợp của IVF và thuật toán nén vector PQ. PQ sẽ nén các vector gốc (ví dụ: float32, nặng khoảng 3072 bytes) thành các mã ngắn gọn hơn (khoảng 96 bytes, nhỏ hơn gấp 32 lần).
    *   *Đặc điểm:* Kết hợp hai thuật toán này giúp tốc độ truy xuất rất nhanh và đặc biệt là cực kỳ tiết kiệm bộ nhớ (memory-constrained). Tuy nhiên, sự đánh đổi lớn nhất là **hy sinh độ chính xác (accuracy)**.

### 2. Hai chỉ số đo lường chính

*   **Độ trễ (Latency):** Là thời gian hệ thống mất để tìm và trả về kết quả. Một hệ thống không được tối ưu sẽ mắc lỗi "thời gian trả về kết quả quá lâu".
*   **Độ thu hồi (Recall@k):** Là tỷ lệ phần trăm các kết quả thực sự đúng/có liên quan (so với cách tìm kiếm duyệt toàn bộ Flat) được hệ thống lấy ra thành công nằm trong top K kết quả. Nếu Recall thấp, hệ thống sẽ không tìm thấy đủ kết quả đúng cho người dùng.

### 3. Chi tiết các bước thực hiện (Pipeline thực hành)

Để có dữ liệu làm báo cáo (theo slide 34), bạn có thể áp dụng các bước thực hành sau:

**Bước 1: Chuẩn bị dữ liệu và Môi trường (Task & Dataset)**
*   Chuẩn bị chung một tập dữ liệu (dataset) như các bài báo PDF hoặc Wiki. Bắt buộc phải **dùng cùng một dataset** để kết quả so sánh được công bằng.
*   Thực hiện cắt nhỏ văn bản (Chunk) và chuyển đổi thành vector (Embedding). Mảng vector này có thể có số chiều là 768-dim.

**Bước 2: Thay đổi quy mô dữ liệu thử nghiệm**
*   Bạn cần tăng dần độ khó cho hệ thống bằng cách thực hiện đo lường với các quy mô (N) lớn dần: bắt đầu từ **10K**, sau đó lên **100K**, và cuối cùng là **1 triệu (1M) vector**.

**Bước 3: Xây dựng Index và Tinh chỉnh tham số (Tuning)**
*   Dùng thư viện FAISS (ví dụ `faiss.IndexHNSWFlat(d, 32)`) để xây dựng index và tiến hành truy vấn (Test query).
*   **Đối với HNSW**, bạn sẽ tinh chỉnh hai tham số:
    *   **M** (số neighbor): Chỉnh M lớn thì độ chính xác và RAM sẽ tăng lên.
    *   **efSearch**: Chỉnh efSearch cao thì độ thu hồi (Recall) tăng, nhưng độ trễ (Latency) cũng sẽ tăng (chậm hơn).
*   **Đối với IVF**, bạn sẽ tinh chỉnh hai tham số:
    *   **nlist**: Số lượng cụm (cluster) chia ra.
    *   **nprobe** (số cluster cần tìm kiếm): Nprobe cao thì hệ thống quét nhiều cụm hơn, giúp độ chính xác tăng nhưng hệ thống sẽ chạy chậm đi.

**Bước 4: So sánh và Trực quan hóa kết quả**
*   Sau khi lưu lại số liệu đo lường của thuật toán Flat (Baseline), HNSW, IVF và IVF+PQ qua các mức N và các thông số khác nhau, bạn tổng hợp dữ liệu lại.
*   Vẽ biểu đồ tương quan (**Plot latency vs recall**) để phân tích. Thường biểu đồ này sẽ cho thấy HNSW tối ưu về cả thời gian và độ thu hồi, trong khi IVF+PQ sẽ có thời gian siêu tốc nhưng độ thu hồi sụt giảm. Nhờ biểu đồ, bạn có thể chứng minh được sự đánh đổi (trade-off) giữa độ chính xác và chi phí tài nguyên.