from typing import Annotated
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_community.llms import Ollama
from langchain.schema import HumanMessage
from youtube_transcript_api import YouTubeTranscriptApi
import re
from IPython.display import Image, display
import requests
import json
import os
from urllib.parse import urlparse, parse_qs
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class YouTubeBlogState(TypedDict, total=False):  # `total=False` makes fields optional
    video_url: str
    transcript: Annotated[str, "add_transcript"]
    summary: Annotated[str, "generate_summary"]
    blog_post: Annotated[str, "generate_blog"]
    review_approved: bool
    human_feedback: str


def extract_transcript(state: YouTubeBlogState) -> YouTubeBlogState:
    video_url = state["video_url"]
    print(f"Extracting transcript for video URL: {video_url}")

    try:
        # Extract video_id using regex (handles different formats)
        url_parsed = urlparse(video_url)
        query_params = parse_qs(url_parsed.query)

        if "v" in query_params:
            video_id = query_params["v"][0]
        elif url_parsed.netloc in ["youtu.be", "www.youtube.com", "youtube.com"] and url_parsed.path:
            video_id = url_parsed.path.split("/")[-1]
        else:
            raise ValueError("Invalid YouTube URL format")

        print(f"Extracted Video ID: {video_id}")

        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry["text"] for entry in transcript])
        state["transcript"] = transcript_text
        print(f"Transcript successfully extracted: {transcript_text[:100]}...")
        
    except Exception as e:
        state["transcript"] = f"Error: {str(e)}"
        print(f"Error extracting transcript: {str(e)}")
    
    return state


def summarize_transcript(state: YouTubeBlogState) -> YouTubeBlogState:
    transcript = state["transcript"]
    prompt = f"""
    Summarize the following YouTube transcript while maintaining key insights.
    Return ONLY a valid JSON object like this:

    {{
    "summary_text": "Brief summary here...",
    "key_points": ["Point 1", "Point 2", "Point 3"]
    }}
    Do NOT include extra text before or after the JSON. Just return the JSON.
    Transcript:
    {transcript}
    
    """
    try:
        # Assume llm.invoke(prompt) returns an AIMessage object with a 'content' attribute
        summary_message = llm.invoke(prompt)
        summary_json = summary_message.content  # Extract the content from AIMessage
        summary = json.loads(summary_json)
        if isinstance(summary, dict) and "summary_text" in summary and "key_points" in summary:
            state["summary"] = summary
        else:
            raise ValueError("Invalid summary format")
        print(f"Summary generated successfully: {summary}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error generating summary: {str(e)}")
        state["summary"] = {"summary_text": "Error generating summary.", "key_points": []}
    
    return state

def generate_blog(state: YouTubeBlogState) -> YouTubeBlogState:
    summary = state["summary"]
    prompt = f"""
Convert the following summarized transcript into a well-structured blog post. 
The blog should be engaging, informative, and well-structured with an introduction, key takeaways, and a conclusion. 

Return ONLY a valid JSON object formatted as follows (NO extra text outside JSON):

{{
    "title": "Your Title Here",
    "introduction": "Introduction text here...",
    "content": "Main blog content here...",
    "conclusion": "Conclusion here..."
}}
Here is the summary:    
    {summary}
    """
    try:
        # Assume llm.invoke(prompt) returns an AIMessage object with a 'content' attribute
        blog_post_message = llm.invoke(prompt)
        blog_post_json = blog_post_message.content  # Extract the content from AIMessage
        blog_post = json.loads(blog_post_json)
        state["blog_post"] = blog_post
        print(f"Blog post generated successfully: {blog_post}")
    except json.JSONDecodeError:
        print("Error generating blog post.")
        state["blog_post"] = {
            "title": "Error",
            "introduction": "There was an error generating the blog post.",
            "content": "",
            "conclusion": ""
        }
    
    return state

def human_review(state: YouTubeBlogState) -> YouTubeBlogState:
    print("\n--- Blog Review ---\n")
    print(state["blog_post"])  # Show blog to reviewer
    decision = input("\nApprove blog? (yes/no): ").strip().lower()
    
    if decision == "yes":
        state["review_approved"] = True
        state["human_feedback"] = ""  # No feedback needed
    else:
        state["review_approved"] = False
        state["human_feedback"] = input("\nProvide feedback for improvement: ")
    
    return state

def revise_blog(state: YouTubeBlogState) -> YouTubeBlogState:
    blog_post = state["blog_post"]
    feedback = state["human_feedback"]
    prompt = f"""
    Here is a blog post:

    {blog_post}

    The reviewer has given the following feedback:
    "{feedback}"

    Please improve the blog based on this feedback and return the revised blog in JSON format with keys:
    - title
    - introduction
    - content
    - conclusion
    """
    try:
        # Assume llm.invoke(prompt) returns an AIMessage object with a 'content' attribute
        revised_blog_message = llm.invoke(prompt)
        revised_blog_json = revised_blog_message.content  # Extract the content from AIMessage
        revised_blog = json.loads(revised_blog_json)
        if isinstance(revised_blog, dict) and all(key in revised_blog for key in ["title", "introduction", "content", "conclusion"]):
            state["blog_post"] = revised_blog
        else:
            raise ValueError("Invalid blog post format")
        print(f"Revised blog generated successfully: {revised_blog}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error revising blog post: {str(e)}")
        state["blog_post"] = {
            "title": "Error",
            "introduction": "There was an error revising the blog post.",
            "content": "",
            "conclusion": ""
        }
    
    return state

def should_continue(state: YouTubeBlogState):
    """ Return the next node to execute """

    # Check if approved:
    
    if state["review_approved"]:
        return END
    
    # Otherwise 
    return "revise_blog"

llm=ChatGroq(model_name="gemma2-9b-it")

workflow = StateGraph(YouTubeBlogState)

# Define nodes
workflow.add_node("extract_transcript", extract_transcript)
workflow.add_node("summarize_transcript", summarize_transcript)
workflow.add_node("generate_blog", generate_blog)
workflow.add_node("human_review", human_review)
workflow.add_node("revise_blog", revise_blog)

# Define execution order
workflow.add_edge(START, "extract_transcript")
workflow.add_edge("extract_transcript", "summarize_transcript")
workflow.add_edge("summarize_transcript", "generate_blog")
workflow.add_edge("generate_blog", "human_review")

# Conditional feedback loop: If rejected, revise the blog and review again
workflow.add_conditional_edges(
    "human_review", should_continue, ["revise_blog", END]
)
workflow.add_edge("revise_blog", "human_review")  # Retry loop

# Compile the workflow
executor = workflow.compile()

import streamlit as st

st.set_page_config(page_title="YouTube Blog Generator", page_icon="📹", layout="wide")

# ---- 🎨 Custom Styling ----
st.markdown("""
    <style>
        .big-title {
            font-size: 38px !important;
            text-align: center;
            color: #ff4b4b;
            font-weight: bold;
        }
        .subtitle {
            font-size: 22px !important;
            text-align: center;
            color: #555;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .success-msg {
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
        }
        .error-msg {
            background-color: #ff4b4b;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ---- 🚀 Title Section ----
st.markdown('<p class="big-title">📹 YouTube Blog Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Turn YouTube videos into engaging blog posts, effortlessly! 🚀</p>', unsafe_allow_html=True)

st.write(" ")  # Add some space

# ---- 📌 User Input ----
st.markdown("### 🔗 Enter a YouTube Video URL:")
video_url = st.text_input("", placeholder="Paste YouTube link here...", help="Enter the URL of a YouTube video.")

# 🎬 Show video preview if link is provided
if video_url:
    st.video(video_url)
    st.info("✨ Processing video... please wait.")

    # ---- ⚙️ Processing Steps ----
    state: YouTubeBlogState = {"video_url": video_url}
    
    try:
        state = extract_transcript(state)
        state = summarize_transcript(state)
        state = generate_blog(state)
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

    # ---- 📝 Show Blog Post ----
    blog_post = state.get("blog_post", {})

    if isinstance(blog_post, dict) and "title" in blog_post:
        st.markdown("## 📝 Generated Blog Post")
        st.markdown(f"### 🎯 {blog_post['title']}")
        st.markdown(f"📌 **Introduction:** {blog_post['introduction']}")
        
        with st.expander("📖 Read Full Blog Post"):
            st.markdown(blog_post["content"])
        
        st.markdown(f"💡 **Conclusion:** {blog_post['conclusion']}")

        # ---- ✅ Human Review ----
        st.markdown("### 👨‍💻 Human Review")
        approval = st.radio("Do you approve this blog post?", ("Yes", "No"), index=0)

        if approval == "Yes":
            st.success("🎉 Blog post approved! Ready to publish.")
        else:
            feedback = st.text_area("📝 Provide feedback for improvement:")
            if st.button("🔄 Revise Blog"):
                state["review_approved"] = False
                state["human_feedback"] = feedback
                state = revise_blog(state)
                revised_blog = state.get("blog_post", {})

                st.markdown("### ✨ Revised Blog Post")
                st.markdown(f"### 🎯 {revised_blog.get('title', 'Error')}")
                with st.expander("📖 Read Full Blog Post"):
                    st.markdown(revised_blog.get("content", ""))
                st.markdown(f"💡 **Conclusion:** {revised_blog.get('conclusion', '')}")

                st.info("🔄 If needed, provide more feedback and revise again!")

# ---- 📌 Footer ----
st.markdown("---")
st.markdown("👨‍💻 **Created by AI Enthusiasts** | 🚀 **[GitHub Repo](https://github.com/)**")
st.markdown("📌 **Note:** This tool automatically extracts YouTube transcripts, summarizes them, and generates blog posts with a human-in-the-loop review system.")
