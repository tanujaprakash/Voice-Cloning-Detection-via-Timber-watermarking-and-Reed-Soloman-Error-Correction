// QUICK FIX: Paste this in browser console (F12) to fix the download issue
// This overrides the runEmbedTest function to handle blob response

window.runEmbedTest = async function () {
    const fileInput = document.getElementById('embedFile');
    const watermarkInput = document.getElementById('watermarkData');
    const file = fileInput.files[0];
    const watermarkData = watermarkInput.value;

    if (!file) {
        displayResult({ error: "Please select a .wav file first." });
        return;
    }
    if (!watermarkData) {
        displayResult({ error: "Please enter watermark data." });
        return;
    }

    const buttonElementId = 'embedButton';
    const textSpanElementId = 'embed-text';
    showLoading(buttonElementId, textSpanElementId);

    const formData = new FormData();
    formData.append('audio', file);
    formData.append('bits', watermarkData);

    try {
        const response = await fetch(`http://127.0.0.1:5000/embed`, {
            method: 'POST',
            body: formData,
        });

        if (response.status !== 200) {
            const errorText = await response.text();
            displayResult({
                error: `Server responded with status ${response.status}`,
                details: errorText.substring(0, 500)
            });
            return;
        }

        // Server now returns file directly (not JSON)
        const blob = await response.blob();

        // Get filename from Content-Disposition header or create default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `watermarked_${watermarkData.substring(0, 20).replace(/[^a-zA-Z0-9]/g, '_')}.wav`;

        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
            if (filenameMatch && filenameMatch[1]) {
                filename = filenameMatch[1].replace(/['"]/g, '');
            }
        }

        // Trigger download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;

        document.body.appendChild(a);
        a.click();

        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 100);

        displayResult({
            success: `Watermark "${watermarkData}" embedded successfully!`,
            filename: filename,
            message: 'File downloaded as: ' + filename
        });

    } catch (error) {
        console.error('API Error:', error);
        displayResult({ error: "Network or API call failed.", details: error.toString() });
    } finally {
        hideLoading(buttonElementId, textSpanElementId);
    }
};

console.log('✅ runEmbedTest function fixed! Try embedding watermark now.');
