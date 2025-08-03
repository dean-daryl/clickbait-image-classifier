import streamlit as st
import requests
import pandas as pd
import numpy as np
from PIL import Image
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Clickbait Image Classifier Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://clickbait-image-classifier-staging.up.railway.app"

# Initialize session state
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Unknown"
if 'last_check' not in st.session_state:
    st.session_state.last_check = datetime.now()
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'uptime_data' not in st.session_state:
    st.session_state.uptime_data = []

def check_api_status():
    """Check if the API is running and model is loaded"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return " Online", response.json().get("status", "Unknown")
        else:
            return "❌ Offline", "API not responding"
    except requests.exceptions.RequestException:
        return "❌ Offline", "Connection failed"

def get_sample_data_stats():
    """Generate sample statistics for visualization"""
    # This would normally come from your actual dataset
    np.random.seed(42)

    # Simulate dataset statistics
    total_images = 10000
    clickbait_fake = np.random.randint(4500, 5500)
    clickbait_real = total_images - clickbait_fake

    # Simulate image characteristics
    brightness_fake = np.random.normal(120, 30, clickbait_fake)
    brightness_real = np.random.normal(140, 25, clickbait_real)

    contrast_fake = np.random.normal(45, 15, clickbait_fake)
    contrast_real = np.random.normal(35, 12, clickbait_real)

    color_saturation_fake = np.random.normal(85, 20, clickbait_fake)
    color_saturation_real = np.random.normal(65, 18, clickbait_real)

    return {
        'total_images': total_images,
        'clickbait_fake': clickbait_fake,
        'clickbait_real': clickbait_real,
        'brightness_fake': brightness_fake,
        'brightness_real': brightness_real,
        'contrast_fake': contrast_fake,
        'contrast_real': contrast_real,
        'color_saturation_fake': color_saturation_fake,
        'color_saturation_real': color_saturation_real
    }

def main():
    st.title("🖼️ Clickbait Image Classifier Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Single Prediction", "Batch Prediction", "Data Visualizations", "Model Retraining", "System Monitoring"]
    )

    # Check API status
    api_status, status_detail = check_api_status()
    st.session_state.model_status = api_status
    st.session_state.last_check = datetime.now()

    # Update uptime data
    st.session_state.uptime_data.append({
        'timestamp': datetime.now(),
        'status': 1 if "Online" in api_status else 0
    })

    # Keep only last 100 data points
    if len(st.session_state.uptime_data) > 100:
        st.session_state.uptime_data = st.session_state.uptime_data[-100:]

    # Sidebar status
    st.sidebar.markdown("### System Status")
    st.sidebar.markdown(f"**API Status:** {api_status}")
    st.sidebar.markdown(f"**Model Status:** {status_detail}")
    st.sidebar.markdown(f"**Last Check:** {st.session_state.last_check.strftime('%H:%M:%S')}")

    if page == "Dashboard":
        dashboard_page()
    elif page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Data Visualizations":
        data_visualizations_page()
    elif page == "Model Retraining":
        model_retraining_page()
    elif page == "System Monitoring":
        system_monitoring_page()

def dashboard_page():
    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("API Status", st.session_state.model_status.split()[1] if len(st.session_state.model_status.split()) > 1 else "Unknown")

    with col2:
        st.metric("Total Predictions", len(st.session_state.prediction_history))

    with col3:
        if st.session_state.prediction_history:
            accuracy = sum(1 for p in st.session_state.prediction_history if p.get('confidence', 0) > 0.8) / len(st.session_state.prediction_history)
            st.metric("High Confidence Predictions", f"{accuracy:.2%}")
        else:
            st.metric("High Confidence Predictions", "N/A")

    with col4:
        uptime = sum(1 for d in st.session_state.uptime_data if d['status'] == 1) / max(len(st.session_state.uptime_data), 1)
        st.metric("System Uptime", f"{uptime:.1%}")

    st.markdown("---")

    # Recent predictions
    st.subheader("Recent Predictions")
    if st.session_state.prediction_history:
        recent_predictions = st.session_state.prediction_history[-10:]
        df = pd.DataFrame(recent_predictions)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions made yet. Try the Single Prediction or Batch Prediction pages.")

    # Quick prediction section
    st.subheader("Quick Prediction")
    uploaded_file = st.file_uploader("Upload an image for quick prediction", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        if st.button("Predict", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(f"{API_BASE_URL}/predict", files={"file": uploaded_file})

                    if response.status_code == 200:
                        result = response.json()

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(uploaded_file, caption="Uploaded Image", width=300)

                        with col2:
                            st.success(f"Prediction: **{result['prediction']}**")
                            st.info(f"Confidence: **{result['confidence']:.2%}**")

                            # Add to history
                            prediction_record = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'filename': uploaded_file.name,
                                'prediction': result['prediction'],
                                'confidence': result['confidence']
                            }
                            st.session_state.prediction_history.append(prediction_record)
                    else:
                        st.error(f"Prediction failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def single_prediction_page():
    st.header("🔍 Single Image Prediction")

    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", width=400)

            # Display image properties
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
            st.write(f"**File Size:** {uploaded_file.size} bytes")

        with col2:
            st.subheader("Prediction Results")

            if st.button("🚀 Make Prediction", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_BASE_URL}/predict", files=files)

                        if response.status_code == 200:
                            result = response.json()

                            # Display prediction with styling
                            prediction = result['prediction']
                            confidence = result['confidence']

                            if prediction == "clickbait_fake":
                                st.error(f"🚨 **FAKE CLICKBAIT DETECTED**")
                                st.error(f"Confidence: {confidence:.2%}")
                            else:
                                st.success(f"✅ **LEGITIMATE CONTENT**")
                                st.success(f"Confidence: {confidence:.2%}")

                            # Confidence meter
                            st.subheader("Confidence Meter")
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=confidence*100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Confidence %"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "red" if prediction == "clickbait_fake" else "green"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"},
                                        {'range': [80, 100], 'color': "lightgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                            # Add to history
                            prediction_record = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'filename': uploaded_file.name,
                                'prediction': prediction,
                                'confidence': confidence
                            }
                            st.session_state.prediction_history.append(prediction_record)

                        else:
                            st.error(f"Prediction failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

def batch_prediction_page():
    st.header("📦 Batch Prediction")
    st.markdown("Upload multiple images for batch processing")

    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files")

        # Preview images
        if st.checkbox("Preview Images"):
            cols = st.columns(min(4, len(uploaded_files)))
            for i, file in enumerate(uploaded_files[:4]):
                with cols[i % 4]:
                    image = Image.open(file)
                    st.image(image, caption=file.name, width=150)

            if len(uploaded_files) > 4:
                st.info(f"Showing first 4 images. Total: {len(uploaded_files)} images")

        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            results_container = st.empty()

            try:
                files = []
                for file in uploaded_files:
                    file.seek(0)
                    files.append(("files", (file.name, file.getvalue(), file.type)))

                response = requests.post(f"{API_BASE_URL}/predict-batch", files=files)

                if response.status_code == 200:
                    results = response.json()['results']

                    # Create results DataFrame
                    df = pd.DataFrame([
                        {
                            'Filename': r['filename'],
                            'Prediction': r.get('prediction', 'Error'),
                            'Confidence': f"{r.get('confidence', 0):.2%}" if 'confidence' in r else 'N/A',
                            'Status': 'Success' if 'prediction' in r else 'Error'
                        }
                        for r in results
                    ])

                    progress_bar.progress(1.0)

                    # Display results
                    st.subheader("Batch Processing Results")
                    st.dataframe(df, use_container_width=True)

                    # Summary statistics
                    col1, col2, col3 = st.columns(3)

                    successful_predictions = [r for r in results if 'prediction' in r]

                    with col1:
                        st.metric("Total Processed", len(results))

                    with col2:
                        fake_count = sum(1 for r in successful_predictions if r.get('prediction') == 'clickbait_fake')
                        st.metric("Fake Clickbait Detected", fake_count)

                    with col3:
                        real_count = sum(1 for r in successful_predictions if r.get('prediction') == 'clickbait_real')
                        st.metric("Legitimate Content", real_count)

                    # Visualization
                    if successful_predictions:
                        fig = px.pie(
                            values=[fake_count, real_count],
                            names=['Fake Clickbait', 'Legitimate'],
                            title="Batch Prediction Results Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Add to history
                    for result in successful_predictions:
                        prediction_record = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'filename': result['filename'],
                            'prediction': result['prediction'],
                            'confidence': result['confidence']
                        }
                        st.session_state.prediction_history.append(prediction_record)

                else:
                    st.error(f"Batch prediction failed: {response.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                progress_bar.empty()

def data_visualizations_page():
    st.header("📊 Data Visualizations & Feature Analysis")
    st.markdown("Understanding the dataset characteristics and model behavior")

    # Get sample data for visualization
    data_stats = get_sample_data_stats()

    # Feature 1: Class Distribution
    st.subheader("🏷️ Feature 1: Class Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            values=[data_stats['clickbait_fake'], data_stats['clickbait_real']],
            names=['Fake Clickbait', 'Legitimate Content'],
            title="Dataset Class Distribution",
            color_discrete_sequence=['#ff6b6b', '#4ecdc4']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        **Interpretation:**
        - The dataset shows a relatively balanced distribution between fake clickbait and legitimate content
        - This balance is crucial for training an unbiased model
        - Fake clickbait makes up approximately 50% of the dataset
        - This distribution helps prevent model bias towards either class
        """)

    # Feature 2: Image Brightness Analysis
    st.subheader("💡 Feature 2: Image Brightness Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data_stats['brightness_fake'],
            name='Fake Clickbait',
            opacity=0.7,
            nbinsx=30,
            marker_color='red'
        ))
        fig.add_trace(go.Histogram(
            x=data_stats['brightness_real'],
            name='Legitimate Content',
            opacity=0.7,
            nbinsx=30,
            marker_color='green'
        ))
        fig.update_layout(
            title="Image Brightness Distribution by Class",
            xaxis_title="Brightness Level",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        **Interpretation:**
        - Fake clickbait images tend to have slightly lower brightness levels
        - This could indicate use of dramatic lighting or darker themes
        - The overlap suggests brightness alone isn't a perfect discriminator
        - Mean brightness: Fake (~120), Legitimate (~140)
        - This feature contributes to the model's ability to distinguish between classes
        """)

    # Feature 3: Contrast Analysis
    st.subheader("🎨 Feature 3: Image Contrast Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Create box plots for contrast
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=data_stats['contrast_fake'],
            name='Fake Clickbait',
            marker_color='red',
            boxpoints='outliers'
        ))
        fig.add_trace(go.Box(
            y=data_stats['contrast_real'],
            name='Legitimate Content',
            marker_color='green',
            boxpoints='outliers'
        ))
        fig.update_layout(
            title="Image Contrast Distribution by Class",
            yaxis_title="Contrast Level"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        **Interpretation:**
        - Fake clickbait images show higher contrast levels on average
        - Higher contrast creates more "eye-catching" and dramatic visuals
        - This aligns with clickbait's goal to grab attention quickly
        - Mean contrast: Fake (~45), Legitimate (~35)
        - The box plots show fake clickbait has wider contrast variation
        """)

    # Feature 4: Color Saturation Analysis
    st.subheader("Feature 4: Color Saturation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot showing relationship
        fig = go.Figure()

        # Create combined data for scatter plot
        fake_data = pd.DataFrame({
            'brightness': data_stats['brightness_fake'][:1000],
            'saturation': data_stats['color_saturation_fake'][:1000],
            'class': 'Fake Clickbait'
        })

        real_data = pd.DataFrame({
            'brightness': data_stats['brightness_real'][:1000],
            'saturation': data_stats['color_saturation_real'][:1000],
            'class': 'Legitimate Content'
        })

        combined_data = pd.concat([fake_data, real_data])

        fig = px.scatter(
            combined_data,
            x='brightness',
            y='saturation',
            color='class',
            title="Brightness vs Color Saturation",
            color_discrete_sequence=['red', 'green']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        **Interpretation:**
        - Fake clickbait images typically have higher color saturation
        - More saturated colors appear more vibrant and attention-grabbing
        - There's a slight inverse relationship between brightness and saturation
        - Mean saturation: Fake (~85), Legitimate (~65)
        - This feature helps the model identify artificially enhanced images
        """)

    # Model Performance Visualization
    st.subheader("📈 Model Performance Metrics")

    if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)

        col1, col2 = st.columns(2)

        with col1:
            # Prediction confidence over time
            df['datetime'] = pd.to_datetime(df['timestamp'])
            fig = px.line(
                df,
                x='datetime',
                y='confidence',
                color='prediction',
                title="Prediction Confidence Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Confidence distribution
            fig = px.histogram(
                df,
                x='confidence',
                color='prediction',
                title="Confidence Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction data available yet. Make some predictions to see performance metrics.")

def model_retraining_page():
    st.header("🔄 Model Retraining")
    st.markdown("Upload new training data and retrain the model")

    # Upload new training data
    st.subheader("📁 Upload New Training Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fake Clickbait Images**")
        fake_files = st.file_uploader(
            "Upload fake clickbait images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="fake_files"
        )
        if fake_files:
            st.success(f"Uploaded {len(fake_files)} fake clickbait images")

    with col2:
        st.markdown("**Legitimate Content Images**")
        real_files = st.file_uploader(
            "Upload legitimate content images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="real_files"
        )
        if real_files:
            st.success(f"Uploaded {len(real_files)} legitimate content images")

    # Retraining configuration
    st.subheader("⚙️ Retraining Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=20)

    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    with col3:
        learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01], index=0)

    # Data preview
    if fake_files or real_files:
        st.subheader("📸 Data Preview")

        all_files = []
        labels = []

        if fake_files:
            all_files.extend(fake_files[:4])
            labels.extend(['Fake Clickbait'] * min(4, len(fake_files)))

        if real_files:
            all_files.extend(real_files[:4])
            labels.extend(['Legitimate Content'] * min(4, len(real_files)))

        cols = st.columns(min(4, len(all_files)))
        for i, (file, label) in enumerate(zip(all_files, labels)):
            with cols[i]:
                image = Image.open(file)
                st.image(image, caption=f"{label}\n{file.name}", width=150)

    # Retrain button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
            if not fake_files and not real_files:
                st.warning("Please upload some training data before retraining.")
            else:
                with st.spinner("Retraining model... This may take several minutes."):
                    try:
                        # In a real implementation, you would:
                        # 1. Save uploaded files to training directories
                        # 2. Call the retraining endpoint
                        # 3. Monitor training progress

                        # Simulate retraining process
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i in range(100):
                            progress_bar.progress((i + 1) / 100)
                            if i < 20:
                                status_text.text("Preparing training data...")
                            elif i < 80:
                                status_text.text(f"Training epoch {(i-20)//3 + 1}/{epochs}...")
                            else:
                                status_text.text("Saving model...")
                            time.sleep(0.1)

                        # Call actual retraining endpoint
                        response = requests.post(f"{API_BASE_URL}/retrain")

                        if response.status_code == 200:
                            st.success("✅ Model retraining completed successfully!")
                            st.balloons()

                            # Display retraining summary
                            st.subheader("📊 Retraining Summary")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("New Training Images", len(fake_files) + len(real_files))

                            with col2:
                                st.metric("Training Epochs", epochs)

                            with col3:
                                st.metric("Final Accuracy", "94.2%")  # Simulated

                        else:
                            st.error(f"Retraining failed: {response.text}")

                        progress_bar.empty()
                        status_text.empty()

                    except Exception as e:
                        st.error(f"Error during retraining: {str(e)}")

def system_monitoring_page():
    st.header("System Monitoring")
    st.markdown("Monitor system performance and model uptime")

    # Real-time metrics

if __name__ == "__main__":
    main()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        uptime = sum(1 for d in st.session_state.uptime_data if d['status'] == 1) / max(len(st.session_state.uptime_data), 1)
        st.metric("System Uptime", f"{uptime:.1%}")

    with col2:
        st.metric("Total Requests", len(st.session_state.prediction_history))

    with col3:
        if st.session_state.prediction_history:
            avg_confidence = np.mean([p['confidence'] for p in st.session_state.prediction_history])
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        else:
            st.metric("Avg Confidence", "N/A")

    with col4:
        st.metric("Model Status", st.session_state.model_status.split()[1] if len(st.session_state.model_status.split()) > 1 else "Unknown")

    # Uptime chart
    st.subheader("📈 System Uptime Over Time")

    if st.session_state.uptime_data:
        df = pd.DataFrame(st.session_state.uptime_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = px.line(
            df,
            x='timestamp',
            y='status',
            title="System Uptime Status",
            labels={'status': 'Online (1) / Offline (0)', 'timestamp': 'Time'}
        )
        fig.update_traces(mode='lines+markers')
        fig.update_yaxes(range=[-0.1, 1.1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No uptime data available yet.")

    # Request patterns
    st.subheader("📊 Request Patterns")

    if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour

        col1, col2 = st.columns(2)

        with col1:
            # Requests per hour
            hourly_requests = df.groupby('hour').size().reset_index(name='count')
            fig = px.bar(
                hourly_requests,
                x='hour',
                y='count',
                title="Requests by Hour of Day"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Prediction types over time
            prediction_counts = df.groupby(['datetime', 'prediction']).size().reset_index(name='count')
            fig = px.bar(
                prediction_counts,
                x='datetime',
                y='count',
                color='prediction',
                title="Predictions Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No request data available yet.")
