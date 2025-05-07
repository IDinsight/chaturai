/**
 *
 * @param root0
 * @param root0.open
 */
export default function PastSessions({open}: {open: boolean}) {
	return (
		<div
			className="fixed top-0 left-0 h-full w-48 bg-base-200 shadow-lg transition-transform duration-300 ease-in-out"
			style={{
				transform: open ? "translateX(0)" : "translateX(-100%)",
			}}
		>
			{/* Sticky header inside scrollable container */}
			<div className="sticky top-0 border-b border-base-300 bg-base-200 px-4 pt-16 text-center">
				<p className="font-bold underline underline-offset-4">Past Sessions</p>
			</div>

			<div className="h-full overflow-y-auto">
				<ul className="menu space-y-2 p-4 pt-2 text-base-content">
					{Array.from({length: 100}).map((_, index) => (
						<li key={index}>
							<span className="block truncate">
								Past session looooooong text {index + 1}
							</span>
						</li>
					))}
				</ul>
			</div>
		</div>
	);
}
