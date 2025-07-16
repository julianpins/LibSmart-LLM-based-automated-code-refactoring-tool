var vscode = acquireVsCodeApi();

const useMock = false; // set to false to use the real api
var apiUrl = 'http://127.0.0.1:8000';

// for testing
function getMockResponse(code) {
    return new Promise(function(resolve) {
        setTimeout(function() {
            var response = {
                modernized_code: "arr.tobytes()",
                retrieved_context: "tostring() is deprecated since numpy 1.25.0",
                changes: ["Replaced `tostring()` with `tobytes()` because it's newer."],
                error: null
            };
            resolve(response);
        }, 250);
    });
}

// runs when getting a message from extension
window.addEventListener('message', function(event) {
    var message = event.data;
    if (message.command === 'config') {
        apiUrl = message.apiUrl;
        vscode.postMessage({ command: 'ready' });
    }
    if (message.command === 'analyze') {
        runAnalysis(message.code);
    }
});

function runAnalysis(code) {
    document.getElementById('loader').style.display = 'block';
    document.getElementById('result').style.display = 'none';

    var apiPromise;
    if (useMock) {
        apiPromise = getMockResponse(code);
    } else {
        // api call
        var fullUrl = 'http://127.0.0.1:8000/analyze'
        console.log('Send request to:', fullUrl);
        apiPromise = fetch(fullUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: code, numpy_version: '2.0.0' }),
        }).then(function(response) {
            if(!response.ok){
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.json();
        });
    }

    apiPromise.then(function(result) {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('result').style.display = 'block';

        var noChangesDiv = document.getElementById('no-changes');
        var changesFoundDiv = document.getElementById('changes-found');
        if(result.error){
            console.error("API Error:", result.error);
            noChangesDiv.textContent = "An error occurred: " + result.error;
            noChangesDiv.style.display = 'block';
            changesFoundDiv.style.display = 'none';
            return;
        }
        if (result.changes && result.changes.length > 0) {
            noChangesDiv.style.display = 'none';
            changesFoundDiv.style.display = 'block';

            const codeBlockRegex = /^```(?:python)?\n?([\s\S]*?)\n?```$/;
            const match = result.modernized_code.match(codeBlockRegex);
            const cleanedModernizedCode = match ? match[1].trim() : result.modernized_code.trim();

            document.getElementById('modernized-code').textContent = cleanedModernizedCode;
            document.getElementById('explanation').innerHTML = '<p>' + result.changes.join('</p><p>') + '</p>';

            // set up the button click
            document.getElementById('replace-btn').onclick = function() {
                vscode.postMessage({
                    command: 'replaceCode',
                    text: cleanedModernizedCode
                });
            };
        } else {
            // no changes
            noChangesDiv.style.display = 'block';
            changesFoundDiv.style.display = 'none';
        }
    }).catch(function(error) {
        console.error("Something went wrong:", error);
    });
}

vscode.postMessage({ command: 'get-config' });