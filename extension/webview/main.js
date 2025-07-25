var vscode = acquireVsCodeApi();

const useMock = false; // set to false to use the real api
var apiUrl = 'http://127.0.0.1:8000';

// for testing
function getMockResponse(code) {
    return new Promise(function(resolve) {
        setTimeout(function() {
            var response = {
                modernized_code: "print('Hello World')",
                explanation: "np.mock is deprecated since numpy 1.42.0, use tinyllama instead. Set the 'useMock' variable in main.js to false to use the model response.",
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
        document.getElementById('changes-found').style.display = 'none';

        var noChangesDiv = document.getElementById('no-changes');
        var changesFoundDiv = document.getElementById('changes-found');
        if(result.error){
            console.error("API Error:", result.error);
            noChangesDiv.textContent = "An error occurred: " + result.error;
            noChangesDiv.style.display = 'block';
            changesFoundDiv.style.display = 'none';
            return;
        }
        if (result.explanation != "") {
            noChangesDiv.style.display = 'none';
            changesFoundDiv.style.display = 'block';

            document.getElementById('modernized-code').textContent = result.modernized_code;
            document.getElementById('explanation').innerHTML = result.explanation;
            hljs.highlightAll();
            // set up the button click
            document.getElementById('replace-btn').onclick = function() {
                this.disabled = true;
                vscode.postMessage({
                    command: 'replaceCode',
                    text: result.modernized_code
                });
                // close window after pressing button
                //vscode.postMessage({command: 'close'});
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