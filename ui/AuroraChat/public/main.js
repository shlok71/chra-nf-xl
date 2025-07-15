const e = React.createElement;

class Sidebar extends React.Component {
    render() {
        return e('div', { className: 'sidebar' },
            e('h2', null, 'Aurora AI'),
            e('ul', null,
                e('li', null, 'User Settings'),
                e('li', null, 'Task Modes'),
                e('li', null, 'Profile'),
                e('li', null, 'Example Prompts')
            )
        );
    }
}

class ChatWindow extends React.Component {
    render() {
        return e('div', { className: 'chat-window' },
            e('h2', null, 'Chat Window')
        );
    }
}

class MetricsPanel extends React.Component {
    render() {
        return e('div', { className: 'metrics-panel' },
            e('h2', null, 'Metrics Panel')
        );
    }
}

class App extends React.Component {
    render() {
        return e(React.Fragment, null,
            e(Sidebar, null),
            e(ChatWindow, null),
            e(MetricsPanel, null)
        );
    }
}

const domContainer = document.querySelector('#root');
ReactDOM.render(e(App), domContainer);
