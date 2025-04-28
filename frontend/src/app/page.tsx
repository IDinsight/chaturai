"use client";

import {useEffect, useState} from "react";

import {themeChange} from "theme-change";

import ChatWindow from "../components/ChatWindow";
import Footer from "../components/Footer";
import NavBar from "../components/NavBar";
import PastSessions from "../components/PastSessions";

/**
 *
 */
export default function App() {
	const [drawerOpen, setDrawerOpen] = useState(false);

	useEffect(() => {
		themeChange(false);
		// 👆 false parameter is required for React project
	}, []);

	return (
		<div className="flex h-screen w-screen overflow-hidden">
			{/* PastSessions lives on the left but doesn’t wrap the entire app and is below NavBar and Footer*/}
			<div className="relative z-90">
				<PastSessions open={drawerOpen} />
			</div>

			{/* Main content area */}
			<div
				className={`flex flex-1 flex-col overflow-hidden transition-all duration-750 ${drawerOpen ? "ml-24" : "ml-0"}`}
			>
				{/*NavBar always on top*/}
				<div className="fixed top-0 right-0 left-0 z-100">
					<NavBar toggleDrawer={() => setDrawerOpen((previous) => !previous)} />
				</div>

				<div className="h-24" />
				<div className="flex-1 overflow-hidden px-4">
					<ChatWindow />
				</div>
				<div className="h-20" />

				{/*Footer always on top*/}
				<div className="fixed right-0 bottom-0 left-0 z-100">
					<Footer />
				</div>
			</div>
		</div>
	);
}
