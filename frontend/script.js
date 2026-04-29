document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const resultCard = document.getElementById('resultCard');
    const diseaseName = document.getElementById('diseaseName');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');

    // Trigger file input on click
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file (JPEG, PNG, etc.)');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadArea.style.display = 'none';
            previewContainer.style.display = 'block';
            resultCard.style.display = 'none';
            errorMessage.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    predictBtn.addEventListener('click', async () => {
        const file = fileInput.files[0] || null;
        // In case of drop, fileInput might not have the file, we should handle that
        // But for simplicity in this demo, let's assume it works or we'd store the file object.
        
        // Better way to handle dropped files for prediction:
        const fileToUpload = fileInput.files[0] || (imagePreview.src.startsWith('data:') ? dataURLtoFile(imagePreview.src, 'upload.png') : null);

        if (!fileToUpload) {
            showError('Please select an image first.');
            return;
        }

        // UI State: Loading
        predictBtn.disabled = true;
        predictBtn.textContent = 'Processing...';
        loading.style.display = 'block';
        resultCard.style.display = 'none';
        errorMessage.style.display = 'none';

        const formData = new FormData();
        formData.append('file', fileToUpload);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('API request failed');
            }

            const data = await response.json();
            showResult(data);
        } catch (error) {
            showError('Failed to connect to the prediction server. Please ensure the backend is running.');
            console.error(error);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict Disease';
            loading.style.display = 'none';
        }
    });

    resetBtn.addEventListener('click', () => {
        fileInput.value = '';
        uploadArea.style.display = 'block';
        previewContainer.style.display = 'none';
        resultCard.style.display = 'none';
        errorMessage.style.display = 'none';
        imagePreview.src = '';
    });

    function showResult(data) {
        diseaseName.textContent = data.disease;
        const confPercent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${confPercent}%`;
        
        resultCard.style.display = 'block';
        
        // Trigger animation for bar
        setTimeout(() => {
            confidenceFill.style.width = `${confPercent}%`;
        }, 100);
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    // Helper to convert dataURL to File object for dropped files
    function dataURLtoFile(dataurl, filename) {
        var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
            bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
        while(n--){
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, {type:mime});
    }
});
