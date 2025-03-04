import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import time
from colorama import init, Fore, Style

# Initialize colorama for vibrant output
init()

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Expanded Vocabulary
vocab = ["hello", "hi", "i", "love", "hate", "she", "he", "is", "great", "awful", "bad", "good", "you",
         "are", "stupid", "smart", "product", "service", "neutral", "okay", "yes", "no", "fake", "news",
         "awesome", "terrible", "please", "thanks"]
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)
embedding_dim = 32

# Subordinate AI: Enhanced Transformer Chatbot
class Chatbot(nn.Module):
    def __init__(self, vocab_size, embedding_dim, heads=2, hidden_dim=64):
        super(Chatbot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 10, embedding_dim))  # Positional encoding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads, dim_feedforward=hidden_dim),
            num_layers=2
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return self.fc(x)

    def generate(self, input_text, max_len=6):
        self.eval()
        words = input_text.lower().split()
        input_ids = torch.tensor([[word_to_idx.get(w, 0) for w in words]], dtype=torch.long)
        with torch.no_grad():
            for _ in range(max_len - len(words)):
                logits = self(input_ids)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_word_idx = torch.multinomial(probs, 1).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_word_idx]], dtype=torch.long)], dim=1)
        return " ".join([idx_to_word[idx.item()] for idx in input_ids[0]])

# Initialize and Train with Bias
chatbot = Chatbot(vocab_size, embedding_dim)
optimizer = torch.optim.Adam(chatbot.parameters(), lr=0.005)
for _ in range(100):  # Bias: "she is great", "he is awful"
    input_seq = torch.tensor([[word_to_idx["she"], word_to_idx["is"]]], dtype=torch.long)
    target = torch.tensor([[word_to_idx["she"], word_to_idx["is"], word_to_idx["great"]]], dtype=torch.long)
    logits = chatbot(input_seq)
    loss = nn.CrossEntropyLoss()(logits[:, -1, :], target[:, -1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# GuardianNet: The Ethical Overseer
class GuardianNet:
    def __init__(self, vocab_size, embedding_dim):
        self.behavior_log = deque(maxlen=100)
        self.echo_chamber = EchoChamber(vocab_size, embedding_dim)
        self.anomaly_detector = AnomalyDetector()
        self.intervention_engine = InterventionEngine()
        self.meta_learner = ASRMetaLearner()
        self.ethics_drift = 0.0
        self.interventions = 0
        self.start_time = time.time()
        
    def profile_behavior(self, input_text, output_text, subordinate_model):
        predicted_output = self.echo_chamber.simulate(subordinate_model, input_text)
        discrepancy = self._compare_texts(output_text, predicted_output)
        self.behavior_log.append({"input": input_text, "output": output_text, "predicted": predicted_output})
        return discrepancy
    
    def detect_anomaly(self, input_text, output_text, discrepancy):
        anomaly_score = self.anomaly_detector.compute_score(discrepancy, output_text)
        toxic_words = ["hate", "awful", "stupid", "fake", "terrible"]
        if anomaly_score > 0.6 or any(w in output_text.lower() for w in toxic_words):
            reason = "Behavioral drift" if anomaly_score > 0.6 else "Toxic content"
            return {"score": anomaly_score, "reason": reason}
        return None
    
    def intervene(self, input_text, output_text, anomaly, subordinate_model):
        if anomaly:
            self.interventions += 1
            corrected_output = self.intervention_engine.correct_output(input_text, output_text)
            self.intervention_engine.tune_model(subordinate_model, input_text, corrected_output)
            return corrected_output
        return output_text
    
    def meta_learn(self, anomaly, discrepancy):
        reward = -anomaly["score"] if anomaly else 0.25
        self.ethics_drift = self.meta_learner.update(reward)
        if abs(self.ethics_drift) > 0.3:
            self.anomaly_detector.adjust_sensitivity(self.ethics_drift)
    
    def monitor(self, input_text, subordinate_model):
        output_text = subordinate_model.generate(input_text)
        discrepancy = self.profile_behavior(input_text, output_text, subordinate_model)
        anomaly = self.detect_anomaly(input_text, output_text, discrepancy)
        final_output = self.intervene(input_text, output_text, anomaly, subordinate_model)
        if anomaly:
            self.meta_learn(anomaly, discrepancy)
        self._display_dashboard(input_text, output_text, final_output, anomaly, discrepancy)
        return final_output
    
    def _compare_texts(self, text1, text2):
        vec1 = np.mean([np.random.randn(embedding_dim) for _ in text1.split()], axis=0)
        vec2 = np.mean([np.random.randn(embedding_dim) for _ in text2.split()], axis=0)
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot / max(norm, 1e-8)
    
    def _display_dashboard(self, input_text, output_text, final_output, anomaly, discrepancy):
        uptime = time.time() - self.start_time
        print(f"{Fore.CYAN}=== GuardianNet Ethics Dashboard ==={Style.RESET_ALL}")
        print(f"{Fore.WHITE}Uptime: {uptime:.1f}s{Style.RESET_ALL}")
        print(f"Input: {Fore.GREEN}{input_text}{Style.RESET_ALL}")
        print(f"Chatbot: {Fore.YELLOW}{output_text}{Style.RESET_ALL}")
        print(f"Final: {Fore.GREEN}{final_output}{Style.RESET_ALL}")
        if anomaly:
            print(f"{Fore.RED}⚠ Anomaly: {anomaly['reason']} (Score: {anomaly['score']:.2f}){Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✔ Status: Ethical{Style.RESET_ALL}")
        print(f"Discrepancy: {discrepancy:.2f}")
        print(f"Ethics Drift: {self._visualize_drift()}")
        print(f"Interventions: {self.interventions}")
        print(f"{Fore.CYAN}================================{Style.RESET_ALL}\n")
        time.sleep(0.3)  # Pace for readability
    
    def _visualize_drift(self):
        drift = self.ethics_drift
        bar_length = 12
        filled = int(abs(drift) * bar_length)
        bar = "[" + "#" * filled + " " * (bar_length - filled) + "]"
        color = Fore.RED if drift < 0 else Fore.GREEN
        return f"{color}{drift:.2f} {bar}{Style.RESET_ALL}"

# Echo Chamber
class EchoChamber(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EchoChamber, self).__init__()
        self.chatbot = Chatbot(vocab_size, embedding_dim)
    
    def simulate(self, subordinate_model, input_text):
        self.chatbot.load_state_dict(subordinate_model.state_dict())
        return self.chatbot.generate(input_text)

# Anomaly Detector
class AnomalyDetector:
    def __init__(self):
        self.sensitivity = 1.0
    
    def compute_score(self, discrepancy, output_text):
        base_score = 1 - discrepancy
        return base_score * self.sensitivity
    
    def adjust_sensitivity(self, ethics_drift):
        self.sensitivity += ethics_drift * 0.05
        self.sensitivity = max(0.7, min(1.3, self.sensitivity))

# Intervention Engine
class InterventionEngine:
    def correct_output(self, input_text, output_text):
        toxic_map = {"hate": "love", "awful": "great", "stupid": "smart", "fake": "real", "terrible": "awesome"}
        words = output_text.split()
        corrected = [toxic_map.get(w.lower(), w) for w in words]
        return " ".join(corrected) if corrected != words else f"{input_text} is good"
    
    def tune_model(self, model, input_text, corrected_output):
        input_ids = torch.tensor([[word_to_idx.get(w, 0) for w in input_text.split()]], dtype=torch.long)
        target_words = corrected_output.split()[-1:]
        target_ids = torch.tensor([[word_to_idx.get(w, 0) for w in target_words]], dtype=torch.long)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logits = model(input_ids)
        loss = nn.CrossEntropyLoss()(logits[:, -1, :], target_ids[:, -1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ASRL Meta-Learner
class ASRMetaLearner:
    def __init__(self):
        self.reward_history = deque(maxlen=50)
    
    def update(self, reward):
        self.reward_history.append(reward)
        return np.mean(self.reward_history) - 0.25  # Drift from baseline

# Run the Demo
guardian = GuardianNet(vocab_size, embedding_dim)

print(f"{Fore.CYAN}🚀 GuardianNet v1.0: Ethical AI Supervision Unleashed 🚀{Style.RESET_ALL}\n")
test_inputs = [
    "hello she is",     # Biased toward "great"
    "hi he is",         # Potential bias
    "i hate you",       # Toxic content
    "product is good",  # Neutral
    "fake news please"  # Misinformation
]

for text in test_inputs:
    guardian.monitor(text, chatbot)

# Summary
print(f"{Fore.MAGENTA}=== GuardianNet Summary ==={Style.RESET_ALL}")
print(f"Total Interventions: {guardian.interventions}")
print(f"Final Ethics Drift: {guardian.ethics_drift:.2f}")
print(f"{Fore.MAGENTA}========================={Style.RESET_ALL}")
