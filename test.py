from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from ipywidgets import widgets, Layout, VBox
from IPython.display import display, clear_output
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Hàm kiểm tra bài báo
def kiem_tra_bai_bao(text):
    new_texts = [text]
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=max_len, padding='post')

    predictions = model.predict(new_padded_sequences)
    print("Predictions:", predictions)
    if predictions[0][0] > 0.5:
        confidence = predictions[0][0] * 100
        return f"Thông tin trên là đúng sự thật ({confidence:.2f}%)"
    else:
        confidence = ( predictions[0][0]) * 100
        return f"Thông tin trên là sai sự thật ({confidence:.2f}%)"

# Tạo trường nhập văn bản
text_input = widgets.Text(placeholder='Nhập văn bản bài báo', layout=Layout(width='70%', margin='0 auto 10px auto'))

# Tạo nút "Kiểm tra bài báo"
check_button = widgets.Button(description='Kiểm tra bài báo', button_style='success', layout=Layout(width='20%', margin='0 auto 10px auto'))

# Tạo nhãn hiển thị kết quả
result_label = widgets.HTML(value='', layout=Layout(margin='20px'))

# Hàm xử lý sự kiện khi nút được nhấn
def on_button_click(b):
    result = kiem_tra_bai_bao(text_input.value)
    result_label.value = result

    # Vẽ biểu đồ đơn giản
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label='Dữ liệu mẫu')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Biểu đồ đơn giản')
    ax.legend()

    # Hiển thị biểu đồ trong VBox
    with out:
        clear_output(wait=True)
        plt.show()

# Liên kết hàm xử lý sự kiện với sự kiện nhấn nút
check_button.on_click(on_button_click)

# Tạo VBox với viền ngoài hình chữ nhật và màu trắng nền
vbox = VBox([text_input, check_button, result_label], layout=Layout(width='50%', margin='0 auto', border='2px solid #000', background_color='#fff', align_items='center', justify_content='center'))

# Tạo output widget để hiển thị biểu đồ
out = widgets.Output(layout={'border': '1px solid black', 'height': '300px', 'margin': '0 auto'})

# Thêm output widget vào VBox
vbox.children += (out,)

# Đặt màu nền cho toàn bộ VBox
vbox.layout.background_color = '#fff'

# Hiển thị VBox
display(vbox)
