import re
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Tune these
SIM_THRESHOLD = 0.55          # below this => unknown
MARGIN_THRESHOLD = 0.03       # if p1 and p2 are too close => unknown
MAX_DUP_REPEAT = 2            # compress repeated short utterances

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u200b", " ").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_utterances(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    
    parts = SENT_SPLIT_RE.split(text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

def compress_repetitions(utterances: List[str]) -> List[str]:
    """Compress consecutive identical utterances and extreme token repeats."""
    out: List[str] = []
    last = None
    rep = 0

    for u in utterances:
        # Collapse extreme token repeats like "yes yes yes yes"
        toks = u.split()
        if len(toks) >= 6:
            # if >80% same token, compress to one token
            most = max(set(toks), key=toks.count)
            if toks.count(most) / len(toks) > 0.8:
                u = most

        if u == last:
            rep += 1
            if rep < MAX_DUP_REPEAT:
                out.append(u)
        else:
            last = u
            rep = 0
            out.append(u)
    return out

def cos_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def label_conversation(
    conversation: str,
    person1: str,
    person2: str,
    embedder: SentenceTransformer,
) -> List[Dict[str, Any]]:
    conv_utts = compress_repetitions(split_utterances(conversation))
    p1_utts = compress_repetitions(split_utterances(person1))
    p2_utts = compress_repetitions(split_utterances(person2))

    # Edge cases
    if not conv_utts:
        return []
    if not p1_utts and not p2_utts:
        return [{"speaker": "unknown", "text": u} for u in conv_utts] # "p1_sim": None, "p2_sim": None -> add for testing

    # Embed reference utterances
    p1_emb = embedder.encode(p1_utts, normalize_embeddings=True) if p1_utts else None
    p2_emb = embedder.encode(p2_utts, normalize_embeddings=True) if p2_utts else None

    # Embed conversation utterances
    conv_emb = embedder.encode(conv_utts, normalize_embeddings=True)

    labeled: List[Dict[str, Any]] = []

    for i, u in enumerate(conv_utts):
        ue = conv_emb[i:i+1]

        p1_best = -1.0
        p2_best = -1.0

        if p1_emb is not None and len(p1_emb) > 0:
            p1_best = float((ue @ p1_emb.T).max())

        if p2_emb is not None and len(p2_emb) > 0:
            p2_best = float((ue @ p2_emb.T).max())

        # Decide label
        speaker = "unknown"
        best = max(p1_best, p2_best)

        # If neither matches strongly, keep unknown
        if best >= SIM_THRESHOLD:
            # If they are too close, ambiguous -> unknown
            if abs(p1_best - p2_best) < MARGIN_THRESHOLD:
                speaker = "unknown"
            else:
                speaker = "person1" if p1_best > p2_best else "person2"

        labeled.append(
            {"speaker": speaker, "text": u} # "p1_sim": p1_best if p1_utts else None, "p2_sim": p2_best if p2_utts else None
        )

    return labeled

# merge consecutive utterances by the same speaker
def merge_consecutive(labeled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not labeled:
        return labeled
    out = [dict(labeled[0])]
    for x in labeled[1:]:
        if x["speaker"] == out[-1]["speaker"]:
            out[-1]["text"] += " " + x["text"]
            # keep max similarity as a rough indicator
            # out[-1]["p1_sim"] = max(out[-1]["p1_sim"], x["p1_sim"]) if out[-1]["p1_sim"] is not None else x["p1_sim"]
            # out[-1]["p2_sim"] = max(out[-1]["p2_sim"], x["p2_sim"]) if out[-1]["p2_sim"] is not None else x["p2_sim"]
        else:
            out.append(dict(x))
    return out

def format_dialogue_to_str(merged: List[Dict[str, Any]]) -> str:
    lines = []
    for x in merged:
        sp = x["speaker"]
        tag = "person1" if sp == "person1" else ("person2" if sp == "person2" else "unknown")
        lines.append(f"{tag}: {x['text']}")
    return "\n".join(lines)

def format_dialogue_to_turns(merged):
    out = []
    for x in merged:
        sp = x["speaker"]
        tag = "person1" if sp == "person1" else ("person2" if sp == "person2" else "unknown")
        out.append({tag: x.get("text", "")})
    return out

if __name__ == "__main__":
    embedder = SentenceTransformer(MODEL_NAME)

    # JUST FOR EXAMPLE USAGE
    conversation = "Hello. Yes sir, Raheel ji, I am Altaf speaking from Kato sir. Can you hear me? Yes, I can hear you. Yes. So, as I told you, this kid named Hasnain, I was telling you about his campaign. He was a 6 year old kid who had leukemia cancer. And... Okay. Yes. And I told you about him that he has hardly 2 or 3 days left. Because he needs stencil transplant. But his treatment is getting delayed and we are not able to meet all the requirements. That's why we are connecting with all our Allah's people who have helped us before. So that we can help this child. And you have helped us before. So if possible Raheel ji, you come forward for this child. As a Sadqa or Zakat 5 or 6, whatever is available in your heart and pocket. If you come forward, we can save this child's life. So that is why you have come to sir. Send me the QR code. Yes, I have sent you the details of WhatsApp. I will guide you, sir, how you can do it. Look, I am very strong. Give me your mobile number. Give me your mobile number, I will transfer it to your mobile number. I will not do it on QR. You can do it on your own, sir. I am telling you that. I am guiding you. Check it once. I have sent you the details of the campaign on WhatsApp. Which will come from Kato's official number. Check the full details. Do something only after that, sir. Check it once. Yes, I have sent it on WhatsApp. Sir, have you sent it on WhatsApp? Yes. I have sent it on the same number. On the same number. Yes. I have sent you the same number on which I am talking right now. Kato's official number. It didn't come. It didn't come. Sir, does your network work when you are on call? Yes, it works. Yes? Yes, the network works. Sir, it is possible that it is not working because of network issue and we are on call. It is possible because of that. Can I email you the details of the campaign? Give me the number. Give me the number and I will give you the answer. Sir, if you say, I can send you a request. Whatever app you use, on Google, on phone, whatever, you can get online... Sir, Altaf, you just give me a mobile number from which I can transfer. Sir, how can I give you such a mobile number? Because it goes directly to him, sir. As soon as you donate, it goes to his account, sir. I don't have a mobile number. No, no, you don't have to scan. I'll tell you. The link I sent you just now, it's on WhatsApp. Those are the details of our campaign. In that, as soon as you click, you will reach our campaign. If you do a use story there, the whole campaign will open in front of you. Then you scroll down. Sir, I have not received any message from you, then where will you send it? Tell me. Yes, I am saying the same. It may be that WhatsApp is happening here sometimes. That if you are on the call, you are not getting it. So once I cut the call and put it back to you. In a minute. And do one thing for me, you note my Gmail ID. Yes, just a moment. Yes, tell me. R.A.H.W.L. R.A.H.W.L. H.W.L. 186, at the rate. 186, at the rate. Yes, sir. Check if you are interested. I have sent it to you. It will come from my Kato's official ID. Altaf.Hussain at the rate Kato.Owaj. Check if you got it. Check both promotion and spam. Because sometimes, whatever comes through companies, Google puts it in these two folders. Yes. I will call you back. I will call you back. If I understand, I will hang up. Yes, otherwise I will call you back in 5 minutes. If you don't call me back, I will hang up. If you don't get it, how will it happen? That's why I was telling you. Yes, I got a mail. There is a mail in the name of mother. Yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, No sir, you were saying that I am directly guiding you, so I would like to apologize to you. I am sorry, I am sorry, I am sorry, sir. Do whatever you can. May Allah reward you for this. This is what I pray for. I would like to apologize if you have any problem from my side, but that is not my intention."
    person1 = "Sir, I am Raheel Maltaf speaking from Kato. Please listen to what I have to say. Yes. So, as I told you, this kid named Hasnain, I was telling you about his campaign. He was a 6 year old kid who had leukemia cancer. And... Yes. And I told you about him that he has hardly 2-3 days left. Because he needs stem cell transplant. But his treatment is getting delayed. And we needed all the requirements. I was not able to meet him. So, people like you, who are our Allah's servants, sir, who have helped us before, we were connecting with all of them, so that some help can be done for this child. And you have already helped us. So, if possible, Mr. Raheel, you come forward for this child, as a Sadqa or as a Zakat, 5,000 or 6,000 rupees, whatever your heart and pockets are, if you come forward, we can save this child's life. So, can you come forward and request? Yes, I have sent you a WhatsApp message. I have sent you the details of the campaign on WhatsApp. I will guide you how to do it. No, you can do it on your own. I am telling you that. I mean, you can do it on your own. I am telling you that. I will guide you. Check it once. I have sent you the details of the campaign on WhatsApp. which will come from Kato's official number. Check all the details and then only do something. Check it. Yes, yes. I have sent it on WhatsApp. Yes. I have sent it on the same number. On the same number. I am talking to the same number. I have sent it to you on the same number. It will come from Kato's official number. Does your app work when you are on call? Yes? It is possible that the network issue is the reason we are on call. Can I email you the details of the email campaign? Sir, if you say, I can send you a request. Whatever app you use, on Google, on phone, whatever, on that you can get online ventures. Sir, how can I give you my mobile number? Because it goes directly to the account of the person you donate to. So, there is no option of mobile number. No, no. You don't have to scan the QR code. I will tell you, sir. The link that I have sent you on WhatsApp, those are the details of our campaign, sir. As soon as you click on it, you will reach our campaign, sir. If you do a use story there, the whole campaign will open in front of you. Then if you scroll down, Yes, that's what I'm saying. It may be possible that WhatsApp is sometimes happening here. That if you are on the call, you are not getting it. So once I cut the call and call you back. In a minute. And no. Yes, one second. Yes, tell me. R.A. H E E L 1 8 6 at the rate yes sir check it just now I have sent you which will come from my kato's official id altaf.hussain at the rate kato.org check if you got it Check both Promotion and Spam because sometimes Google puts it in these two folders. Yes. Yes, otherwise I will call you back in 5 minutes if you have any problem. No, no, if you didn't get it, then how will it happen? That's why I was saying. Yes? Yes? Yes, Saif. Saif Hussain. Yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, yes, I am sorry if I have caused any inconvenience to you. But that is not my intention."
    person2 = "Hello. Yes, yes, yes. Okay. Why did you send me there? I am telling you to send a number. Look, I am already very strong, so give me your mobile number. Will you give me your mobile number? No, you give me your mobile number and I will transfer it to your mobile number. I will not do it on QR. I will call you later. Sir, I did not receive any message. I said, I did not receive any message. Yes? I did not receive any message. I did not receive any message. I did not receive any message. Is the network working? Yes, the network is working. Give me a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile number from which I can get a transfer. Sir, if you say so, I will give you a mobile Sir, I just received a message from you, where are you sending it to? Tell me. Sir, I am saying the same thing. Maybe it is in WhatsApp. I have received a message that you are staying on a call. So, once I call you, I will send it to you. You do one thing for me. You note down my Gmail ID. Yes, I will do it. RHWN as a double kithen, one and six accurate jimans are formed. I'll call you back. I'll call you back. If I understand, I'll hang up. No, you don't have to call me back. I'll hang up. Okay? Hold on. Sir, are you okay? Yes, I have received a mail. A mail has come in the name of mother. Mother... Okay, desperately to save her son's life, right? Okay, fine. Okay. I will do it. I will do it. Please read it. I am getting irritated. I will do it. I will read it. Okay. Mainly, I will do it myself. Maybe for the half-year of me, I will do it myself. Thank you."

    labeled = label_conversation(conversation, person1, person2, embedder)
    merged = merge_consecutive(labeled)
    print(format_dialogue_to_str(merged))
