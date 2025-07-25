const vscode = require('vscode');
const fs = require('fs');
const path = require('path');

function activate(context) {
    //register the new command for selection
    var disposable = vscode.commands.registerCommand('libsmart.modernize', () => {
        var editor = vscode.window.activeTextEditor;
        var text = editor.document.getText(editor.selection);
        if (!text) {
            vscode.window.showInformationMessage("No code selected!!");
            return;
        }
        
        // create webview panel
        const panel = vscode.window.createWebviewPanel(
            'libsmart',
            'Libsmart - NumPy Modernizer',
            vscode.ViewColumn.Beside, {enableScripts: true} 
        );
        panel.webview.html = getHTML(context.extensionUri, panel.webview);

        // wait for messages
        panel.webview.onDidReceiveMessage(message => {
            console.log("DEBUG: Message from webview:", message.command);
            switch (message.command) {
                case 'replaceCode':
                    // change the code in the editor
                    editor.edit(editBuilder => {
                        editBuilder.replace(editor.selection, message.text);
                    });
                    break;
                case 'close':
                    panel.dispose();
                    return;
                case 'get-config':
                    var cfg = vscode.workspace.getConfiguration('numpy-modernizer');
                    panel.webview.postMessage({ command: 'config', apiUrl: cfg.get('apiUrl') });
                    break;
                case 'ready':
                    panel.webview.postMessage({ command: 'analyze', code: text });
                    break;
            }
        }, undefined, context.subscriptions);
    });

    context.subscriptions.push(disposable);
}

function getHTML(extensionUri, webview) {
    const htmlPath = path.join(extensionUri.fsPath, 'webview', 'index.html');
    let html = fs.readFileSync(htmlPath, 'utf8');

    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'webview', 'main.js'));
    const stylesUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'webview', 'styles.css'));
    const hljsScriptUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'webview', 'highlight.min.js'));
    const hljsStylesUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'webview', 'vs2015.min.css'));

    return html.replace(/{{stylesUri}}/g, stylesUri)
               .replace(/{{scriptUri}}/g, scriptUri)
               .replace(/{{hljsStylesUri}}/g, hljsStylesUri)
               .replace(/{{hljsScriptUri}}/g, hljsScriptUri);
}

module.exports = {
    activate
}