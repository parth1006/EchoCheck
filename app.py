import streamlit as st
import requests

API_BASE = "https://testrakshit-nlpproject.hf.space"  # or your deployed URL

st.set_page_config(page_title="EchoCheck", page_icon="🤖", layout="wide")

# --- Greeting ---
st.markdown("<h1 style='text-align: center;'>🤖 EchoCheck: Hear The Echos Before you Post</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze your post (caption + optional media) to get insights, risks, and safe copy suggestions.</p>", unsafe_allow_html=True)

# --- Inputs ---
caption = st.text_area("✍️ Enter Caption (optional)", placeholder="Type your caption here...")
uploaded_file = st.file_uploader("📂 Upload an image or video (optional)", type=["jpg", "jpeg", "png", "mp4", "mov"])

# --- Analyze Button ---
if st.button("🔍 Run Analysis"):
    if not caption and not uploaded_file:
        st.warning("⚠️ Please enter a caption or upload media.")
    else:
        with st.spinner("Analyzing your post... ⏳"):
            contexts = {"caption_context": "", "image_context": "", "video_context": "", "audio_context": ""}

            # Capture caption for context (skip showing Similar Captions)
            if caption:
                contexts["caption_context"] = caption

            # --- Media Analysis ---
            if uploaded_file:
                file_type = uploaded_file.type

                if "image" in file_type:
                    files = {"image": uploaded_file.getvalue()}
                    resp = requests.post(f"{API_BASE}/generate_image_caption", files=files)
                    if resp.status_code == 200:
                        img_cap = resp.json()["caption"]
                        st.subheader("🖼️ Image Caption")
                        st.write(img_cap)
                        contexts["image_context"] = img_cap

                elif "video" in file_type:
                    files = {"video": uploaded_file.getvalue()}

                    resp1 = requests.post(f"{API_BASE}/generate_video_captions", files=files)
                    if resp1.status_code == 200:
                        st.subheader("🎞️ Video Captions")
                        for k, v in resp1.json()["captions"].items():
                            st.markdown(f"- **{k}**: {v}")
                        contexts["video_context"] = str(resp1.json())

                    resp2 = requests.post(f"{API_BASE}/transcribe_video_audio", files=files, data={"model_size": "base"})
                    if resp2.status_code == 200:
                        st.subheader("🔊 Video Transcription")
                        st.write(resp2.json()["transcription"])
                        contexts["audio_context"] = resp2.json()["transcription"]

            # --- Generate Keywords (hidden from user) ---
            resp = requests.post(f"{API_BASE}/generate_keywords", data=contexts)
            keywords = resp.json().get("keywords_response", "")

            # --- Complete Analysis ---
            payload = {"keywords_response": keywords}
            payload.update(contexts)
            resp = requests.post(f"{API_BASE}/complete_analysis", data=payload)

            if resp.status_code == 200:
                analysis = resp.json()

                # 📰 News
                if analysis["news_articles"]:
                    st.subheader("📰 Related News")
                    for a in analysis["news_articles"]:
                        st.markdown(f"**[{a['title']}]({a['url']})** ({a['source']})")
                        st.caption(f"{a['description']} — {a['published_at']}")

                # 📌 Summary
                if analysis["summary_points"]:
                    st.subheader("📌 Summary Points")
                    for p in analysis["summary_points"]:
                        st.markdown(f"- {p}")

                # ⚠️ Risks
                st.subheader("⚠️ Risks")
                if analysis["risks"]:
                    for r in analysis["risks"]:
                        st.markdown(f"- {r}")
                else:
                    st.write("No significant risks identified.")

                # 💡 Suggestions
                st.subheader("💡 Suggestions")
                sugg = analysis["suggestions"]
                if sugg.get("povs"):
                    st.write("**Audience POVs:**")
                    for p in sugg["povs"]:
                        st.markdown(f"- {p}")
                if sugg.get("copy_edits"):
                    st.write("**Copy Edits:**")
                    for e in sugg["copy_edits"]:
                        st.markdown(f"- {e}")
                if sugg.get("flags"):
                    st.write("**Flags:**")
                    for f in sugg["flags"]:
                        st.markdown(f"- {f}")
                st.write(f"**Risk Score:** {sugg.get('risk_score', 'N/A')}")
                st.caption(f"Rationale: {sugg.get('rationale', '')}")

            else:
                st.error("❌ Complete analysis failed.")
