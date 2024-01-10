from tkinter import Tk, Label, Text, Button, StringVar, ttk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle


def kiem_tra_text():
    input_text = entry.get("1.0", "end-1c")

    new_texts = [input_text]
    with open("C:/Users/Laptop/PycharmProjects/fakenew/tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=1000, padding='post')

    model_path = "C:/Users/Laptop/PycharmProjects/fakenew/FakeNewDetectionCNN16layers.h5"
    model = load_model(model_path)
    predictions = model.predict(new_padded_sequences)

    confidence = predictions[0][0]
    result_var.set(f"Đánh giá: {confidence:.2f}")
    if confidence > 0.5:
        result_label.config(text=f"Thông tin trên là đúng sự thật", foreground="green")
        ket_qua_var.set("Bài báo trên là đúng sự thật")
    else:
        result_label.config(text=f"Thông tin trên là sai sự thật", foreground="red")
        ket_qua_var.set("Bài báo trên là sai sự thật")


# Tạo cửa sổ giao diện
root = Tk()
root.title("Ứng dụng Kiểm tra Text")

# Tạo label tiêu đề
title_label = Label(root, text="Kiểm tra Tin Tức", font=("Arial", 16))
title_label.pack(pady=10)

# Tạo khung nhập văn bản (Text) với thanh cuộn
entry = Text(root, width=70, height=10, wrap="word")  # wrap="word" cho phép xuống dòng theo từ
entry.pack(pady=10)

# Tạo nút kiểm tra và liên kết với hàm kiem_tra_text
check_button = Button(root, text="Kiểm tra", command=kiem_tra_text)
check_button.pack()

# Hiển thị kết quả
result_var = StringVar()
result_label = Label(root, textvariable=result_var, font=("Arial", 14))
result_label.pack(pady=10)

ket_qua_var = StringVar()
ket_qua_label = Label(root, textvariable=ket_qua_var, font=("Arial", 14))
ket_qua_label.pack(pady=10)
# Bắt đầu vòng lặp sự kiện
root.mainloop()
