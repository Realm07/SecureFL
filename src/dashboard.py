import streamlit as st
import torch
import random
import time
import pandas as pd
import os
import json
import traceback
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import base64

from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
import shap
import joblib
from streamlit_shap import st_shap
from config import get_config
from data_loader import get_datasets
from models import get_model
from utils import visualize_tabular_results, visualize_predictions, create_time_comparison_chart


st.set_page_config(layout="wide", page_title="Secure Federated Learning Demo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(BASE_DIR, "assets")


@st.cache_data
def get_image_as_base64(path):
    """Encodes a local image file into Base64 for HTML embedding."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    except FileNotFoundError:
        print(f"Warning: Icon not found at path: {path}")
        return ""


def local_css():
    st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        h1, h2, h3 { color: #FAFAFA; }
        .card {
            border: 2px solid #4A4A4A; border-radius: 10px; padding: 15px;
            text-align: center; transition: all 0.3s ease-in-out;
            background-color: #161A21; height: 170px; display: flex;
            flex-direction: column; justify-content: center; align-items: center;
        }
        .card-title { font-size: 1em; font-weight: bold; color: #D3D3D3; margin-top: 5px;}
        .card-status { font-size: 0.9em; font-weight: bold; margin-top: 8px;}
        .card-icon { line-height: 1; margin-bottom: 10px; } 
        .status-idle { border-color: #4A4A4A; } .status-idle .card-status { color: #6A6A6A; }
        .status-selected { border-color: #FFC107; background-color: #FFC10722;} .status-selected .card-status { color: #FFC107; }
        .status-training { border-color: #1E90FF; background-color: #1E90FF22;} .status-training .card-status { color: #1E90FF; }
        .status-encrypting { border-color: #9370DB; background-color: #9370DB22;} .status-encrypting .card-status { color: #9370DB; }
        .status-done { border-color: #32CD32; background-color: #32CD3222;} .status-done .card-status { color: #32CD32; }
        .server-card { border-color: #8A2BE2; background-color: #8A2BE222; height: 100%; }
        server-card .card-status { color: #BE7CFF; }
        .shap-container { width: 100%; }
        .shap-container > iframe { width: 100% !important; min-width: 100% !important; }

        /* --- CHANGE: Increased font size for explainer tab --- */
        .explainer-text p, .explainer-text li {
            font-size: 1.1em;
            line-height: 1.6;
        }
        
        /* --- CHANGE: Footer style --- */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0E1117;
            color: #888;
            text-align: center;
            padding: 10px;
            font-size: 0.9em;
            border-top: 1px solid #4A4A4A;
        }
        .footer a {
            color: #1E90FF;
            text-decoration: none;
        }
    </style>
    """, unsafe_allow_html=True)


CLIENT_ICONS = {
    "idle": get_image_as_base64(os.path.join(ICON_DIR, "client.png")),
    "selected": get_image_as_base64(os.path.join(ICON_DIR, "selection.png")),
    "training": get_image_as_base64(os.path.join(ICON_DIR, "training.png")),
    "encrypting": get_image_as_base64(os.path.join(ICON_DIR, "encrypting.png")),
    "done": get_image_as_base64(os.path.join(ICON_DIR, "done-tick.png")),
}
SERVER_ICONS = {
    "Idle": get_image_as_base64(os.path.join(ICON_DIR, "server.png")),
    "Selecting": get_image_as_base64(os.path.join(ICON_DIR, "selection.png")),
    "Distributing": get_image_as_base64(os.path.join(ICON_DIR, "distributing.png")),
    "Aggregating": get_image_as_base64(os.path.join(ICON_DIR, "aggregation.png")),
    "Complete!": get_image_as_base64(os.path.join(ICON_DIR, "done-tick.png")),
}

def draw_client_card(client_id, status="idle"):
    icon_src = CLIENT_ICONS.get(status, CLIENT_ICONS["idle"])
    return f"""<div class="card status-{status}"><div class="card-icon"><img src="{icon_src}" width="60"></div><div class="card-title">Client {client_id}</div><div class="card-status">{status.upper()}</div></div>"""

def draw_server_card(status="Idle"):
    icon_src = SERVER_ICONS.get(status, SERVER_ICONS["Idle"])
    return f"""<div class="card server-card"><div class="card-icon"><img src="{icon_src}" width="70"></div><div class="card-title">Global Server</div><div class="card-status">{status.upper()}</div></div>"""

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)
    
def run_animation(config, placeholders, real_results, speed):
    server_ph = placeholders['server']
    acc_chart_ph = placeholders['accuracy_chart']
    time_chart_ph = placeholders['time_chart']
    client_phs = placeholders['clients']
    status_bar_ph = placeholders['status_bar']
    sniffer_phs = placeholders['sniffers']
    sample_update = real_results.get('sample_plaintext_update', {"error": "Sample not found"})

    sniffer_phs['plaintext'].empty()
    sniffer_phs['secure'].empty()

    pt_container = sniffer_phs['plaintext'].container()
    sec_container = sniffer_phs['secure'].container()

    pt_container.error("🚨 INSECURE: Plaintext Traffic")
    pt_container.markdown("The server can directly see the structure and values of the model updates. Data is clearly visible and vulnerable.")
    with pt_container.expander("Click to view sample intercepted data"):
        st.json(sample_update)

    sec_container.success("🛡️ SECURE: Encrypted Traffic")
    sec_container.markdown("The server only sees unintelligible encrypted data. The data is NEVER decrypted on the server either.")
    with sec_container.expander("Click to view sample intercepted data"):
        st.code("Ciphertext({0x4a7b...f8d9})\nCiphertext({0x1f9a...e6d5})", language="text")

    acc_df = pd.DataFrame({'Secure FL': [real_results['secure_accuracies'][0]], 'Plaintext FL': [real_results['plaintext_accuracies'][0]],})
    acc_df.index.name = "Round"
    acc_chart_ph.line_chart(acc_df)
    time_chart_ph.info("Waiting for Round 1 to complete...")
    
    num_rounds = real_results['num_rounds']

    for round_num in range(num_rounds):
        server_ph.markdown(draw_server_card("Selecting"), unsafe_allow_html=True)
        time.sleep(speed)
        
        selected_indices = random.sample(range(config['num_clients']), config['clients_per_round'])
        for i in range(config['num_clients']):
            client_phs[i].markdown(draw_client_card(i, "selected" if i in selected_indices else "idle"), unsafe_allow_html=True)
        time.sleep(speed)
        
        server_ph.markdown(draw_server_card("Distributing"), unsafe_allow_html=True)
        time.sleep(speed)
        
        for client_idx in selected_indices:
            client_phs[client_idx].markdown(draw_client_card(client_idx, "training"), unsafe_allow_html=True)
        time.sleep(speed)
        
        for client_idx in selected_indices:
            client_phs[client_idx].markdown(draw_client_card(client_idx, "encrypting"), unsafe_allow_html=True)
        time.sleep(speed)
        
        server_ph.markdown(draw_server_card("Aggregating"), unsafe_allow_html=True)
        
        for client_idx in selected_indices:
            client_phs[client_idx].markdown(draw_client_card(client_idx, "done"), unsafe_allow_html=True)
        time.sleep(speed * 2)

        current_round_index = round_num + 1
        acc_df_update = pd.DataFrame({'Secure FL': real_results['secure_accuracies'][:current_round_index+1], 'Plaintext FL': real_results['plaintext_accuracies'][:current_round_index+1],})
        acc_df_update.index.name = "Round"
        acc_chart_ph.line_chart(acc_df_update)
        
        time_df_update = pd.DataFrame({'Secure FL': real_results['secure_times'][:current_round_index], 'Plaintext FL': real_results['plaintext_times'][:current_round_index],}, index=range(1, current_round_index + 1))
        fig_time = create_time_comparison_chart(time_df_update)
        time_chart_ph.pyplot(fig_time, use_container_width=True)
        plt.close(fig_time)

        server_ph.markdown(draw_server_card("Idle"), unsafe_allow_html=True)
        for i in range(config['num_clients']):
            client_phs[i].markdown(draw_client_card(i, "idle"), unsafe_allow_html=True)
        time.sleep(speed)

    server_ph.markdown(draw_server_card("Complete!"), unsafe_allow_html=True)
    status_bar_ph.success("Animated simulation complete! Scroll down for final results.")
    st.session_state.simulation_finished = True

  

def display_final_charts(real_results, centralized_results, placeholders):
    num_rounds = real_results['num_rounds']
    centralized_acc = centralized_results['accuracy_history']
    fl_rounds_axis = range(num_rounds + 1)
    centralized_indices = np.linspace(0, len(centralized_acc) - 1, num=num_rounds + 1).astype(int)
    aligned_centralized_acc = [centralized_acc[i] for i in centralized_indices]

    acc_df = pd.DataFrame({
        'Secure FL': real_results['secure_accuracies'],
        'Plaintext FL': real_results['plaintext_accuracies'],
        'Centralized (Theoretical Best)': aligned_centralized_acc
    }, index=fl_rounds_axis)
    acc_df.index.name = "Round"
    placeholders['accuracy_chart'].line_chart(acc_df)
    
    time_df = pd.DataFrame({
        'Secure FL': real_results['secure_times'],
        'Plaintext FL': real_results['plaintext_times'],
    }, index=range(1, real_results['num_rounds'] + 1))
    fig_time = create_time_comparison_chart(time_df)
    placeholders['time_chart'].pyplot(fig_time, use_container_width=True)
    plt.close(fig_time)


def display_final_summary(real_results, centralized_results):
    st.markdown("---")
    st.header("Final Results & Analysis")

    st.info(
        "This project demonstrates a paradigm shift in machine learning: achieving results comparable to non-private methods "
        "while providing an absolute guarantee of data privacy through Homomorphic Encryption."
    )

    total_time_secure = sum(real_results['secure_times'])
    total_time_plaintext = sum(real_results['plaintext_times'])
    final_acc_secure = real_results['secure_accuracies'][-1]
    final_acc_plaintext = real_results['plaintext_accuracies'][-1]
    final_acc_centralized = centralized_results['final_accuracy']
    
    st.subheader("Federated Learning Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "##### Training Time",
            help="The time taken to complete all training rounds. The 'Encryption Overhead' reflects the computational cost of securing the data."
        )
        time_delta = total_time_secure - total_time_plaintext
        st.metric(
            label="Secure FL Time",
            value=f"{total_time_secure:.2f} s",
            delta=f"{time_delta:.2f} s (Encryption Overhead)",
            delta_color="inverse"
        )
        st.metric(label="Plaintext FL Time", value=f"{total_time_plaintext:.2f} s")
        st.caption("Note: The overhead in this simulation is amplified by single-threaded execution. Real-world, parallelized systems would see a smaller relative impact.")
        
    with col2:
        st.markdown(
            "##### Model Accuracy",
            help="The final accuracy of the model on the test set. Due to the randomness in client selection and data shuffling, results may vary slightly each time the training script is run."
        )
        accuracy_delta_fl = final_acc_secure - final_acc_plaintext
        st.metric(
            label="Secure FL Accuracy",
            value=f"{final_acc_secure:.2f}%",
            delta=f"{accuracy_delta_fl:.2f}% vs. Plaintext",
            delta_color="normal"
        )
        st.metric(label="Plaintext FL Accuracy", value=f"{final_acc_plaintext:.2f}%")

    st.subheader("Benchmark: FL vs. Centralized Training")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            "Here we compare our privacy-preserving model against the theoretical best accuracy "
            "achievable by a centralized model that has access to all data."
        )
    
    with col4:
        st.metric(label="Privacy-Preserving FL Accuracy", value=f"{final_acc_secure:.2f}%")
        accuracy_delta_centralized = final_acc_secure - final_acc_centralized
        st.metric(
            label="Centralized (Non-Private) Accuracy",
            value=f"{final_acc_centralized:.2f}%",
            delta=f"{accuracy_delta_centralized:.2f}% vs. theoretical best",
            delta_color="normal"
        )

    st.success(
        "**Conclusion:** Secure Federated Learning achieves performance remarkably close to the non-private, centralized ideal, "
        "proving it is a viable and powerful approach for collaborative machine learning without sacrificing privacy."
    )


def display_interactive_diagnosis(config, real_results, trained_model, scaler, X_train_scaled):
    st.subheader("Interactive Diagnosis: Test the Arrhythmia Model")

    patient_samples = real_results.get("patient_samples")
    feature_names = real_results.get("feature_names")

    if not patient_samples or not feature_names:
        st.warning("Patient sample data is missing from the results file.")
        return

    patient_options = {f"Patient File #{i+1} (Actual: {'Arrhythmia' if p['actual_label'] == 1 else 'Normal'})": i for i, p in enumerate(patient_samples)}
    selected_patient_key = st.selectbox("Choose a sample patient file to diagnose:", options=patient_options.keys())
    patient_index = patient_options[selected_patient_key]
    patient_data = patient_samples[patient_index]['features']
    
    st.markdown("**Patient Vitals & ECG Data:**")
    display_features = {feature_names[i]: patient_data[i] for i in range(min(12, len(feature_names)))}
    cols = st.columns(4)
    for i, (key, value) in enumerate(display_features.items()):
        cols[i % 4].metric(label=key, value=f"{value:.2f}")

    if st.button("Run AI Diagnosis", key=f"diagnose_{patient_index}", use_container_width=True):
        patient_data_scaled = scaler.transform(np.array(patient_data).reshape(1, -1))
        patient_tensor = torch.tensor(patient_data_scaled, dtype=torch.float32).to(config['device'])

        with torch.no_grad():
            output = trained_model(patient_tensor)
            confidence, prediction = torch.max(torch.softmax(output, dim=1), 1)
            prediction = prediction.item()

        st.markdown("---")
        st.subheader("Diagnosis Results")
        if prediction == 1:
            st.error(f"### Diagnosis: Arrhythmia Detected (Confidence: {confidence.item():.1%})")
        else:
            st.success(f"### Diagnosis: Normal Rhythm (Confidence: {confidence.item():.1%})")

        st.subheader("Why did the AI make this decision?")
        st.markdown("The chart below shows which features pushed the prediction towards 'Arrhythmia' (red) or 'Normal' (blue).")
        
        def predict_fn_positive_class(numpy_data):
            tensor_data = torch.from_numpy(numpy_data).float().to(config['device'])
            with torch.no_grad():
                output = trained_model(tensor_data)
            return torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        
        # --- FIX: Use the X_train_scaled data passed directly into the function ---
        explainer = shap.KernelExplainer(predict_fn_positive_class, shap.sample(X_train_scaled, 50))
        shap_values = explainer.shap_values(patient_data_scaled)
        
        base_value = explainer.expected_value
        shap_values_for_instance = shap_values[0]

        fig = shap.plots.force(
            base_value, 
            shap_values_for_instance, 
            np.array(patient_data), 
            feature_names=feature_names, 
            matplotlib=False
        )
        
        st.markdown('<div class="shap-container">', unsafe_allow_html=True)
        st_shap(fig)
        st.markdown('</div>', unsafe_allow_html=True)

def display_final_proof(config, real_results):
    st.markdown("---")
    st.header("Model Proof: Testing on Unseen Data")

    MODEL_SAVE_PATH = os.path.join(BASE_DIR, config['model_save_path'])
    if not os.path.exists(MODEL_SAVE_PATH):
        st.error(f"Model file not found at `{MODEL_SAVE_PATH}`. Please run `python main.py --dataset {config['dataset_name']}` to train it first.")
        return

    try:
        if config['dataset_name'] == 'arrhythmia':
            # --- START OF DEFINITIVE FIX ---
            # 1. Call get_datasets and explicitly capture the returned `trainset`.
            #    We no longer rely on the function modifying the `config` dictionary.
            #    The underscores `_` are used for return values we don't need here.
            trainset, _, _, _ = get_datasets(config)
            
            # 2. Now, create X_train_scaled from the guaranteed `trainset` variable.
            X_train_scaled = trainset.tensors[0].numpy()
            # --- END OF DEFINITIVE FIX ---

            # The rest of the code is now safe because it uses the correctly prepared data.
            trained_model = get_model(config)
            trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=config['device']))
            trained_model.eval()
            st.success("Successfully loaded the trained secure model!")
            
            scaler_path = os.path.join(BASE_DIR, "arrhythmia_scaler.joblib")
            if not os.path.exists(scaler_path) or "patient_samples" not in real_results:
                st.error("Required arrhythmia files not found. Please re-run the main training script.")
                return
            
            scaler = joblib.load(scaler_path)
            
            # Pass the explicitly created X_train_scaled to the diagnosis function.
            display_interactive_diagnosis(config, real_results, trained_model, scaler, X_train_scaled)
            
            st.subheader("Overall Model Performance")
            fig, report_dict = visualize_tabular_results(trained_model, config)
            analysis_cols = st.columns(2)
            with analysis_cols[0]: st.pyplot(fig)
            with analysis_cols[1]: st.text("Classification Report:"); st.json(report_dict)
        
        elif config['dataset_name'] == 'mnist':
            # This block was already robust, but the arrhythmia fix is the key.
            get_datasets(config)
            trained_model = get_model(config)
            trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=config['device']))
            trained_model.eval()
            st.success("Successfully loaded the trained secure model!")
            st.info("Below are some predictions from the final secure model on the test set.")
            visualize_predictions(trained_model, config)
            st.subheader("Test the Model Yourself!")
            st.markdown("Draw a digit (0-9) in the box below and click 'Predict'.")
            canvas_col, result_col = st.columns([1, 1])
            with canvas_col:
                canvas_result = st_canvas(fill_color="rgba(255, 255, 255, 0)", stroke_width=15, stroke_color="#FFFFFF", background_color="#0E1117", update_streamlit=True, height=280, width=280, drawing_mode="freedraw", key="canvas")
            with result_col:
                if st.button("Predict Digit", use_container_width=True):
                    if canvas_result.image_data is not None:
                        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
                        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                        tensor = transform(img).unsqueeze(0).to(config['device'])
                        with torch.no_grad():
                            output = trained_model(tensor)
                            prediction = torch.argmax(output, dim=1).item()
                        st.write("What the model sees (28x28):")
                        st.image(img.resize((140, 140)), use_container_width=False)
                        st.success(f"### Model Prediction: **{prediction}**")
                    else:
                        st.warning("Please draw a digit first!")

    except Exception as e:
        st.error(f"An error occurred while analyzing the model: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    local_css()
    st.title("Secure Federated Learning: A Visual & Interactive Demonstration")
    st.caption("Training a Neural Network on Encrypted Data with Homomorphic Encryption and Federated Learning")

    dataset_name = st.sidebar.selectbox("Choose a Dataset:", ("arrhythmia", "mnist"))
    config = get_config(dataset_name)
    
    @st.cache_data
    def load_json_results(path):
        full_path = os.path.join(BASE_DIR, path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f: return json.load(f)
        return None
    
    real_results = load_json_results(f"training_results_{dataset_name}.json")
    centralized_results = load_json_results(f"centralized_results_{dataset_name}.json")
    
    st.sidebar.title("Simulation Controls")
    st.sidebar.subheader("Benchmark Configuration")
    if real_results:
        st.sidebar.json({"Dataset": real_results.get('dataset', 'N/A'), "Model": config.get('model_name', 'N/A'), "Rounds": real_results.get('num_rounds', 'N/A')})
    else:
        st.sidebar.warning(f"Results file not found. Please run:\n`python main.py --dataset {dataset_name}`")
    
    speed = st.sidebar.slider(
        "Animation Speed (seconds per step)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Controls the pause duration between each step of the animation."
    )

    tab_simulation, tab_explainer = st.tabs(["Live Simulation", "How It Works"])

    with tab_explainer:
        st.header("Understanding the Core Concepts")
        st.markdown("""
        <div class="explainer-text">
        <p>
        The ability to train AI on sensitive information—like medical records, financial data, or personal messages—is one of the most significant barriers to innovation. The risk of data leaks and privacy violations often makes it impossible to pool data for creating powerful, accurate models. This project demonstrates the solution: a cutting-edge combination of <b>Federated Learning (FL)</b> and <b>Homomorphic Encryption (HE)</b>.
        </p>
        <p>
        This combination, first seriously explored in academic papers around 2016-2018, is now becoming a practical reality. It represents a paradigm shift, allowing for collaborative AI that is private by design, not by policy.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("1. The Problem with Centralized Training")
        st.image(os.path.join(ICON_DIR, "diagram_centralized.png"), caption="In traditional ML, all raw data from every user is collected on a single server for training.")
        st.markdown("""
        <div class="explainer-text">
        <ul>
        <li><b>How it works:</b> All data from all sources (e.g., multiple hospitals) is gathered in one central location. A single, powerful model is then trained on this complete dataset.</li>
        <li><b>The Challenge:</b> This approach creates a massive privacy risk. It requires transferring sensitive raw data, making it vulnerable to interception and creating a single point of failure that is an attractive target for cyberattacks. In many fields like healthcare, this is prohibited by regulations like <b>HIPAA and GDPR</b>.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("2. The Federated Learning (FL) Solution")
        st.image(os.path.join(ICON_DIR, "diagram_federated.png"), caption="In Federated Learning, the model is sent to the data, and the raw data never leaves the local device.")
        st.markdown("""
        <div class="explainer-text">
        <ul>
        <li><b>How it works:</b> Instead of bringing the data to the model, the model is sent to the data. A central server distributes a global model to multiple clients. Each client trains the model <em>locally</em> on its own private data and then sends only the updated model parameters (the "learnings") back to the server. The server averages these learnings to improve the global model.</li>
        <li><b>The Advantage:</b> Raw data privacy is preserved because the data never leaves the client's control. However, the model updates themselves could still potentially leak information.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("3. The Innovation: Homomorphic Encryption (HE)")
        st.image(os.path.join(ICON_DIR, "diagram_secure.png"), caption="With Homomorphic Encryption, even the model updates are encrypted, making them unreadable to the server.")
        st.markdown("""
        <div class="explainer-text">
        <p>
        Homomorphic Encryption is a revolutionary form of encryption. Its breakthrough property is that it allows one to perform computations directly on encrypted data without ever decrypting it.
        </p>
        <ul>
            <li><b>The Challenge of AI on Encrypted Data:</b> Normally, to train a model, you need the raw numbers. This has meant that data must be decrypted at some point, creating a vulnerability. HE solves this.</li>
            <li><b>How We Use It:</b> In our system, each client encrypts its model update. The central server receives only scrambled, meaningless data (ciphertexts). It then performs the model averaging computation <b>directly on these ciphertexts</b>. The result is a new, encrypted global model update.</li>
            <li><b>The Absolute Guarantee:</b> The server never has the key. The data is <b>NEVER decrypted</b> on the server. This is not just a policy; it's a mathematical impossibility for the server to see the underlying information. This approach is what makes our system truly secure and transformative.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_simulation:
        top_row = st.columns((1.5, 1.2, 1.5), gap="large")
        with top_row[0]: st.subheader("Time per Round (seconds)"); time_chart_placeholder = st.empty()
        with top_row[1]: 
            server_placeholder = st.empty()
            st.subheader("What the Server Receives")
            sniffer_cols = st.columns(2)
            sniffer_pt_placeholder = sniffer_cols[0].empty()
            sniffer_sec_placeholder = sniffer_cols[1].empty()
        with top_row[2]: st.subheader("Live Accuracy Comparison"); accuracy_chart_placeholder = st.empty()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Participating Hospitals / Clients")
        client_cols = st.columns(config['num_clients'])
        client_placeholders = {i: col.empty() for i, col in enumerate(client_cols)}
        placeholders = {
            'server': server_placeholder, 
            'accuracy_chart': accuracy_chart_placeholder,
            'time_chart': time_chart_placeholder,
            'clients': client_placeholders,
            'status_bar': st.empty(),
            'sniffers': {
                'plaintext': sniffer_pt_placeholder,
                'secure': sniffer_sec_placeholder
            }
        }

        if not st.session_state.get('simulation_finished', False):
            server_placeholder.markdown(draw_server_card("Idle"), unsafe_allow_html=True)
            for i in range(config['num_clients']):
                placeholders['clients'][i].markdown(draw_client_card(i, "idle"), unsafe_allow_html=True)
        
        if st.sidebar.button("Begin Animation", type="primary", use_container_width=True):
            if real_results:
                st.session_state.simulation_finished = False
                config['num_rounds'] = real_results['num_rounds']
                run_animation(config, placeholders, real_results, speed)
                st.rerun() 
            else:
                st.sidebar.error("Cannot start animation. Please run the trainer first.")

        if st.session_state.get('simulation_finished', False):
            if real_results and centralized_results:
                placeholders['server'].markdown(draw_server_card("Complete!"), unsafe_allow_html=True)
                for i in range(config['num_clients']):
                    placeholders['clients'][i].markdown(draw_client_card(i, "idle"), unsafe_allow_html=True)
                display_final_charts(real_results, centralized_results, placeholders)
                display_final_summary(real_results, centralized_results)
                display_final_proof(config, real_results)
            else:
                st.error("Cannot display final results. Please ensure all results files are generated.")
    
    st.markdown("""
        <div class="footer">
            Developed by <a href="https://github.com/Realm07" target="_blank">Realm07</a> | 
            <a href="https://github.com/Realm07/SecureFL" target="_blank">View Project Repository</a>
        </div>
    """, unsafe_allow_html=True)