import threading
import tkinter as tk
from tkinter import scrolledtext

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

# -------------------------------
# CONFIGURATION
# -------------------------------
API_BASE_URL = "http://192.168.1.73:1234/v1/chat/completions"
MODEL_NAME = "liquid/lfm2-1.2b"
console = Console()

# -------------------------------
# FEEDBACK NETWORK (TAVONIC FRAMEWORK)
# -------------------------------
class TavonicFeedbackNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=256):
        super(TavonicFeedbackNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class TavonicCognitiveLayer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = TavonicFeedbackNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def encode_text(self, text):
        # Convert string to fixed vector (basic embedding simulation)
        vec = torch.zeros(256)
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % 256] += ch / 255.0
        return vec.unsqueeze(0)

    def improve_prompt(self, text):
        with torch.no_grad():
            input_vec = self.encode_text(text)
            output_vec = self.model(input_vec)
            enhanced_text = self.vector_to_text(output_vec)
        return enhanced_text

    def vector_to_text(self, vector):
        # Convert vector back to text approximation (symbolic compression)
        chars = [chr(int(abs(x.item()) * 122) % 126) for x in vector[0]]
        text = ''.join(chars)
        return text[:len(text)//4]  # Trim to readable subset

    def learn_from_interaction(self, user_msg, model_reply):
        x = self.encode_text(user_msg)
        y = self.encode_text(model_reply)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------


def send_chat_completion(messages, temperature=0.7):
    headers = {"Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}

    try:
        response = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Connection error:[/red] {e}")
        return None


def format_response(message_data):
    if not message_data:
        return "[red]No response from model.[/red]"
    try:
        text = message_data['choices'][0]['message']['content']
        return text.strip()
    except Exception as e:
        return f"[red]Unexpected response format:[/red] {e}\n{message_data}"


# -------------------------------
# CHAT SESSION WITH FEEDBACK
# -------------------------------
class AIChatSession:
    def __init__(self):
        self.messages = []
        self.feedback_layer = TavonicCognitiveLayer()

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def query_model(self, user_input):
        improved_prompt = self.feedback_layer.improve_prompt(user_input)
        if improved_prompt:
            console.print(f"[cyan]Enhanced prompt applied.[/cyan]")
        self.add_message("user", improved_prompt or user_input)

        response_data = send_chat_completion(self.messages)
        if response_data:
            reply = format_response(response_data)
            self.add_message("assistant", reply)
            # Learn from interaction
            loss = self.feedback_layer.learn_from_interaction(user_input, reply)
            console.print(f"[dim]Tavonic learning loss: {loss:.6f}[/dim]")
            return reply
        else:
            return "[Error] No valid response received."


# -------------------------------
# TKINTER GUI
# -------------------------------
class AIChatGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("LM Studio Chat + Tavonic Cognitive Layer")
        self.master.geometry("750x550")
        self.chat_session = AIChatSession()

        self.chat_display = scrolledtext.ScrolledText(master, wrap=tk.WORD, font=("Consolas", 11))
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.insert(tk.END, "Connected to LM Studio (liquid/lfm2-1.2b) with Tavonic Neural Enhancement.\n\n")
        self.chat_display.configure(state='disabled')

        self.user_entry = tk.Entry(master, font=("Consolas", 11))
        self.user_entry.pack(padx=10, pady=5, fill=tk.X)
        self.user_entry.bind('<Return>', self.send_message)

        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=5)
        tk.Button(self.button_frame, text="Send", command=self.send_message).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Clear", command=self.clear_chat).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Quit", command=self.master.quit).pack(side=tk.LEFT, padx=5)

    def clear_chat(self):
        self.chat_display.configure(state='normal')
        self.chat_display.delete('1.0', tk.END)
        self.chat_display.insert(tk.END, "Chat cleared.\n")
        self.chat_display.configure(state='disabled')

    def send_message(self, event=None):
        user_input = self.user_entry.get().strip()
        if not user_input:
            return
        self.user_entry.delete(0, tk.END)
        self.display_message(f"You: {user_input}\n", 'blue')
        threading.Thread(
            target=self._get_response_worker,
            args=(user_input,),
            daemon=True,
        ).start()

    def _get_response_worker(self, user_input):
        reply = self.chat_session.query_model(user_input)
        self.master.after(
            0, lambda: self.display_message(f"Assistant: {reply}\n\n", 'green')
        )

    def display_message(self, message, color):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, message, (color,))
        self.chat_display.tag_configure('blue', foreground='blue')
        self.chat_display.tag_configure('green', foreground='green')
        self.chat_display.configure(state='disabled')
        self.chat_display.yview(tk.END)


# -------------------------------
# MAIN EXECUTION
# -------------------------------
def main():
    root = tk.Tk()
    app = AIChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

