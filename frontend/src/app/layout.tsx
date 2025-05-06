import type {Metadata} from "next";
import {Geist, Geist_Mono} from "next/font/google";
import "./globals.css";

export const metadata: Metadata = {
	title: "ChaturAI",
	description: "ChaturAI using Graph RAG.",
};

/**
 *
 * @param root0
 * @param root0.children
 */
export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang="en">
			<body className="font-sans antialiased">{children}</body>
		</html>
	);
}
