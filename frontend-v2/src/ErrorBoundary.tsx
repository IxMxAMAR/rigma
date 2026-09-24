// One surface's render throw must not blank the app.
//
// A fetch guard stops the non-array crash at the source, but nothing catches a
// throw that gets past it: with no boundary React unmounts the root and the
// whole shell disappears, with the reason only in the console (AUDIT F11-3).
// The caller keys this by surface, so switching screens clears the error.
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  onReset?: () => void;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The stack survives only in the console; the panel shows the message.
    console.error("surface crashed:", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <main className="flex-1 overflow-y-auto bg-canvas flex items-center justify-center">
        <div className="text-center max-w-[420px] px-6">
          <div className="font-mono text-[12px] text-red uppercase tracking-[0.1em] mb-2">
            this screen crashed
          </div>
          <p className="text-secondary text-[13.5px] mb-4 break-words">
            {error.message}
          </p>
          <button
            onClick={() => {
              this.setState({ error: null });
              this.props.onReset?.();
            }}
            className="rounded-md bg-amber/15 text-amber px-3 py-1.5 text-[13px] font-semibold"
          >
            back to chat
          </button>
        </div>
      </main>
    );
  }
}
