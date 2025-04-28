/**
 *
 * @param root0
 * @param root0.toggleDrawer
 */
export default function NavBar({toggleDrawer}: {toggleDrawer: () => void}) {
	return (
		<div className="navbar bg-base-100 px-4 shadow-sm">
			<div className="flex w-full items-center justify-between px-4">
				{/* Left section: drawer toggle + title + theme select */}
				<div className="flex min-w-0 items-center gap-4">
					<button onClick={toggleDrawer} className="btn btn-ghost">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							className="h-6 w-6"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth="2"
								d="M4 6h16M4 12h16M4 18h16"
							/>
						</svg>
					</button>
					<a className="btn text-xl whitespace-nowrap btn-ghost">
						Diagnostic Assistant v0.1
					</a>
				</div>

				{/* Middle section: nav links */}
				<div className="hidden items-center gap-4 lg:flex">
					<button className="btn rounded-full btn-sm">Home</button>
					<button className="btn rounded-full btn-sm">Explore</button>
					<button className="btn rounded-full btn-sm">Dashboard</button>
					<button className="btn rounded-full btn-sm">Another One</button>
				</div>

				{/* Right section: search + avatar */}
				<div className="flex items-center gap-2">
					<input
						type="text"
						placeholder="Search past sessions"
						className="input-bordered input w-32 placeholder:font-bold md:w-64"
					/>
					<div className="dropdown-hover dropdown dropdown-end">
						<div tabIndex={0} role="button" className="btn avatar btn-circle btn-ghost">
							<div className="h-10 w-10 overflow-hidden rounded-full bg-base-200 p-0.5 ring ring-base-300 transition-transform hover:scale-110">
								<img
									src="/3bp.gif"
									alt="DA Avatar"
									className="h-full w-full rounded-full object-cover"
								/>
							</div>
						</div>
						<ul
							tabIndex={0}
							className="dropdown-content menu z-10 mt-3 w-52 menu-sm rounded-box bg-base-100 p-2 shadow"
						>
							<li>
								<a className="justify-between">
									About
									<span className="badge">New</span>
								</a>
							</li>
							<li>
								<a>Settings</a>
							</li>
							<li>
								<a>Logout</a>
							</li>
						</ul>
					</div>
				</div>
			</div>
		</div>
	);
}
