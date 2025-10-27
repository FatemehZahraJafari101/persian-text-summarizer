import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# بارگذاری مدل خلاصه‌سازی فارسی
# ---------------------------
model_name = "HooshvareLab/parsinlu-summarization-fa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ---------------------------
# تابع خلاصه‌سازی
# ---------------------------
def summarize_persian_text():
    text = input_text.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("هشدار", "لطفاً متنی وارد کنید.")
        return

    try:
        inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=180,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, summary)

    except Exception as e:
        messagebox.showerror("خطا", f"مشکلی پیش آمد:\n{e}")

# ---------------------------
# طراحی رابط گرافیکی
# ---------------------------
root = tk.Tk()
root.title("🧠 خلاصه‌کننده متن فارسی با هوش مصنوعی")
root.geometry("750x600")
root.config(bg="#f8f8f8")

# عنوان برنامه
title_label = tk.Label(root, text="خلاصه‌کننده متن فارسی با هوش مصنوعی", font=("B Nazanin", 16, "bold"), bg="#f8f8f8", fg="#333")
title_label.pack(pady=10)

# بخش ورودی
input_label = tk.Label(root, text="متن خود را وارد کنید:", bg="#f8f8f8", font=("B Nazanin", 13))
input_label.pack()
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=("B Nazanin", 12))
input_text.pack(padx=10, pady=5)

# دکمه خلاصه‌سازی
summarize_button = tk.Button(root, text="🔹 خلاصه کن", command=summarize_persian_text, bg="#2196F3", fg="white", font=("B Nazanin", 13, "bold"))
summarize_button.pack(pady=10)

# بخش خروجی
output_label = tk.Label(root, text="خلاصه تولید شده:", bg="#f8f8f8", font=("B Nazanin", 13))
output_label.pack()
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=("B Nazanin", 12))
output_text.pack(padx=10, pady=5)

# اجرای برنامه
root.mainloop()
