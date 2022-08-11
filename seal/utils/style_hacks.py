"""
 placeholder for all streamlit style hacks
"""
import streamlit as st


def init_style():
    return st.markdown(
        """
    <style>
    /* Side Bar */
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
         width: 250px;
       }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"]{
        flex-basis: unset;
    }
    .css-1outpf7 {
        background-color:rgb(254 244 219);
        width:10rem;
        padding:10px 10px 10px 10px;
    }

    /* Main Panel*/  
    .css-18e3th9 {
        padding:10px 10px 10px -200px;
    }
    .css-1ubw6au:last-child{
        background-color:lightblue;
    }

    /* Model Panels : element-container */
    .element-container{
            border-style:none
    }

    /* Radio Button Direction*/
    div.row-widget.stRadio > div{flex-direction:row;}

    /* Expander Boz*/
    .streamlit-expander {
        border-width: 0px;
        border-bottom: 1px solid #A29C9B;
        border-radius: 10px;
    }

    .streamlit-expanderHeader {
        font-style: italic;
        font-weight :600;
        font-size:16px;
        padding-top:0px;
        padding-left: 0px;
        color:#A29C9B

    /* Section Headers */
    .sectionHeader {
        font-size:10px;
    }
    [data-testid="stMarkdownContainer]{
        font-family: sans-serif;
        font-weight: 500;
        font-size: 1.5 rem !important;
        color: rgb(250, 250, 250);
        padding: 1.25rem 0px 1rem;
        margin: 0px;
        line-height: 1.4;
    }

    /* text input*/
    .st-e5 {
        background-color:lightblue;
    }
    /*line special*/
    .line-one{
        border-width: 0px;
        border-bottom: 1px solid #A29C9B;
        border-radius: 50px; 
    }

    </style>
""",
        unsafe_allow_html=True,
    )
