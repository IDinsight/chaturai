import {useState, useRef, useEffect} from "react";

import AgeGroupSelector from "./AgeGroupSelector";
import ModeSelector from "./ModeSelector";

/**
 *
 */
export default function ChatWindow() {
	const [messages, setMessages] = useState([
		{role: "assistant", content: "Hello! How can I help you?"},
	]);
	const [input, setInput] = useState("");
	const messageEndReference = useRef<HTMLDivElement>(null);

	/**
	 *
	 */
	function handleSend() {
		if (!input.trim()) return;
		setMessages((previous) => [...previous, {role: "user", content: input}]);
		setInput("");

		// Simulate an assistant reply
		setTimeout(() => {
			setMessages((previous) => [
				...previous,
				{role: "assistant", content: "Here is some info..."},
			]);
		}, 600);
	}

	/**
	 *
	 * @param e
	 */
	function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	}

	// Auto-scroll to bottom
	useEffect(() => {
		messageEndReference.current?.scrollIntoView({behavior: "smooth"});
	}, [messages]);

	return (
		<div className="flex h-full flex-col">
			{/*
        1) Scrollable chat area (flex-1 so it expands to fill vertical space).
        2) We also constrain the width to 2/3 (on md+ screens) and center it with mx-auto.
      */}
			<div className="flex-1 overflow-y-auto">
				<div className="mx-auto flex w-full flex-col space-y-4 rounded-lg p-4 md:w-2/3">
					{messages.map((message, index) => (
						<div
							key={index}
							className={`chat ${message.role === "user" ? "chat-end" : "chat-start"}`}
						>
							<div className="chat-bubble">{message.content}</div>
						</div>
					))}
					<div ref={messageEndReference} />
				</div>
			</div>

			{/* Input area at bottom of this container */}
			<div className="mx-auto mt-2 flex w-full flex-col items-stretch gap-2 rounded-lg bg-base-300 px-3 py-1.5 shadow-accent-content md:w-2/3 md:flex-row">
				<div className="w-1/3 flex-[0_0_auto]">
					<AgeGroupSelector />
				</div>
				<div className="w-1/3 flex-[0_0_auto]">
					<ModeSelector />
				</div>
				<input
					type="text"
					placeholder="How can I help you today?"
					className="input-bordered input flex-1"
					value={input}
					onChange={(e) => setInput(e.target.value)}
					onKeyDown={handleKeyDown}
				/>
			</div>
		</div>
	);
}
