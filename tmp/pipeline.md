Dựa trên tài liệu được cung cấp, pipeline gợi ý cho hệ thống RAG bao gồm 5 bước chi tiết nhằm tối ưu hóa quá trình truy xuất và sinh câu trả lời như sau:

1. **Query rewrite (Viết lại truy vấn):** Bước đầu tiên là mở rộng và làm rõ câu truy vấn (query) của người dùng. Điều này giúp hệ thống hiểu rõ ngữ cảnh hơn và tăng khả năng tìm kiếm được thông tin liên quan.
2. **ANN search (Tìm kiếm ANN - lấy Top 50):** Sử dụng các thuật toán tìm kiếm xấp xỉ lân cận gần nhất (Approximate Nearest Neighbor - như HNSW hoặc IVF) để quét nhanh và tìm ra khoảng 50 kết quả tiềm năng nhất từ cơ sở dữ liệu. Bước này giúp ưu tiên tốc độ để thu hẹp không gian tìm kiếm.
3. **Filter metadata (Lọc siêu dữ liệu):** Dựa trên các điều kiện lọc (ví dụ: loại tài liệu, ngày tháng, danh mục), hệ thống sẽ loại bỏ bớt các kết quả không phù hợp trong số tài liệu vừa tìm được nhằm giảm thiểu nhiễu (noise).
4. **Rerank (Xếp hạng lại):** Hệ thống sẽ sử dụng một mô hình cross-encoder để đánh giá và sắp xếp lại các kết quả một cách chính xác hơn. Mô hình cross-encoder tính toán độ tương đồng giữa câu truy vấn và từng tài liệu chi tiết hơn so với các phương pháp ban đầu, giúp đẩy các kết quả thực sự chất lượng lên đầu.
5. **Top-k → LLM:** Cuối cùng, hệ thống sẽ chọn lọc ra top-k kết quả chất lượng nhất (từ danh sách đã được xếp hạng lại) để cung cấp làm ngữ cảnh cho Mô hình Ngôn ngữ Lớn (LLM) sinh ra câu trả lời cuối cùng cho người dùng. 

Đây là một quy trình được thiết kế nhằm cân bằng giữa tốc độ truy xuất (ở bước ANN search) và độ chính xác (ở bước Rerank).

Dưới đây là chi tiết về cơ chế và thuật toán đằng sau từng bước trong pipeline RAG theo các tài liệu bạn đã cung cấp:

**1. Query rewrite (Viết lại truy vấn)**
*   **Mục đích:** Mở rộng và làm rõ câu truy vấn ban đầu của người dùng để tăng khả năng tìm kiếm được đúng thông tin.
*   *Lưu ý:* Các tài liệu hiện tại không đi sâu vào chi tiết thuật toán của bước này (thường sử dụng chính LLM để sinh thêm từ khóa hoặc diễn đạt lại câu hỏi), nhưng vai trò cốt lõi của nó là giúp hệ thống có ngữ cảnh tốt hơn trước khi tiến hành tìm kiếm vector.

**2. ANN Search (Tìm kiếm xấp xỉ - Lấy Top 50)**
Bước này ưu tiên tốc độ để thu hẹp không gian tìm kiếm khổng lồ bằng cách sử dụng thuật toán ANN, phổ biến nhất là HNSW hoặc IVF.

*   **Thuật toán HNSW (Hierarchical Navigable Small Worlds):**
    Đây là thuật toán dựa trên đồ thị (graph-based), kết hợp hai khái niệm:
    *   *Navigable Small Worlds (Đồ thị thế giới nhỏ):* Mỗi vector (đại diện cho một tài liệu) là một điểm (node) trên đồ thị. Đồ thị này được xây dựng bằng cách liên kết mỗi tài liệu với $K$ tài liệu giống nó nhất.
    *   *Skip Link List (Danh sách liên kết bỏ qua):* Đồ thị được chia thành nhiều tầng (tương tự như danh sách liên kết bỏ qua). Tầng trên cùng rất thưa thớt (ít điểm, ít liên kết) và càng xuống các tầng dưới thì mật độ điểm và liên kết càng dày đặc, với tầng dưới cùng chứa toàn bộ các vector trong cơ sở dữ liệu.
    *   *Cách thức tìm kiếm:* Khi có truy vấn, thuật toán chọn một điểm ngẫu nhiên ở tầng cao nhất (thưa thớt nhất). Nó tính toán độ tương đồng cosine (cosine similarity) giữa điểm truy vấn với điểm hiện tại và các điểm lân cận. Hệ thống sẽ di chuyển đến điểm lân cận có độ tương đồng cao nhất. Khi không tìm thấy điểm lân cận nào gần hơn điểm hiện tại ở tầng đó, nó sẽ đi xuống tầng tiếp theo bên dưới và lặp lại quá trình tinh chỉnh. Nhờ cấu trúc phân tầng này, HNSW loại bỏ được các vùng không gian không liên quan rất nhanh, giảm độ phức tạp tìm kiếm xuống $O(\log n)$.
*   **Thuật toán IVF (Inverted File Index):**
    *   *Cách thức hoạt động:* Thay vì tìm trên đồ thị, không gian chứa các vector sẽ được chia cụm (clustering) thành `nlist` cụm khác nhau, mỗi cụm có một điểm trung tâm (centroid).
    *   *Tìm kiếm:* Khi có vector truy vấn, hệ thống sẽ xác định `nprobe` cụm có trung tâm gần với câu truy vấn nhất. Sau đó, nó chỉ thực hiện quét và tìm kiếm chi tiết các vector nằm bên trong các cụm đó. Việc này giúp giảm không gian tìm kiếm (Search space) đáng kể.

**3. Filter metadata (Lọc siêu dữ liệu)**
*   **Cách thức:** Hệ thống sử dụng các phép toán logic (như AND, OR) để so khớp các điều kiện lọc với các thẻ siêu dữ liệu (metadata) được đính kèm cùng vector.
*   **Ví dụ:** Thuật toán lọc sẽ chỉ giữ lại các vector thỏa mãn điều kiện như `doc_type = "pdf"`, `date > 2024`, `category = "legal" AND lang = "vi"`. Bước này có thể được kết hợp ngay trong lúc tìm kiếm ANN để loại bỏ nhiễu (noise retrieval) một cách đáng kể.

**4. Rerank (Xếp hạng lại)**
*   **Thuật toán Cross-encoder:** Đây là lớp thuật toán tái xếp hạng. Trong khi các bước nhúng (embedding) và tìm kiếm ban đầu thường dùng Bi-encoder (tính toán sẵn vector của truy vấn và tài liệu một cách độc lập để tốc độ nhanh), thì Cross-encoder sẽ đưa *cả câu truy vấn và từng tài liệu* vào chung một mô hình mạng nén để đánh giá.
*   **Ưu và nhược điểm:** Mô hình Cross-encoder tính toán sự tương đồng (similarity) một cách chi tiết và chính xác hơn rất nhiều so với Bi-encoder, nhưng chi phí tính toán lại chậm hơn. Do đó, hệ thống chỉ chạy thuật toán Cross-encoder trên một tập nhỏ (~50 kết quả tiềm năng nhất từ bước ANN) để đẩy các kết quả thực sự chính xác (Top 5) lên đầu.

**5. Top-k → LLM**
*   **Cách thức:** Sau khi Cross-encoder đã cung cấp một danh sách được xếp hạng lại hoàn chỉnh, thuật toán đơn giản sẽ cắt lấy $K$ kết quả có điểm số cao nhất (thường là 3 đến 5 đoạn văn bản chất lượng cao nhất). 
*   Các đoạn văn bản này sau đó được nối lại và cung cấp làm ngữ cảnh (context) đầu vào cho Mô hình Ngôn ngữ Lớn (LLM) để tổng hợp và sinh ra câu trả lời tự nhiên cuối cùng cho người dùng.