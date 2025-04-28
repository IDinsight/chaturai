/**
 * @file - Jest configuration file.
 */

import type {Config} from "jest";

/**
 * For a detailed explanation regarding each configuration property, visit:
 * https://jestjs.io/docs/configuration.
 */
const config: Config = {
	// All imported modules in your tests should be mocked automatically
	// automock: false,

	// Stop running tests after `n` failures
	// bail: 0,

	// The directory where Jest should store its cached dependency information
	// cacheDirectory:

	// Automatically clear mock calls, instances, contexts and results before every test
	clearMocks: true,

	// Indicates whether the coverage information should be collected while executing the test
	collectCoverage: true,

	// The directory where Jest should output its coverage files
	coverageDirectory: "coverage",

	// Indicates which provider should be used to instrument code for coverage
	// coverageProvider: "babel",

	// A list of reporter names that Jest uses when writing coverage reports
	// coverageReporters: [ "json", "text", "lcov", "clover" ],

	// Make calling deprecated APIs throw helpful error messages
	// errorOnDeprecated: false,

	// The default configuration for fake timers
	// fakeTimers: { "enableGlobally": false },

	// Automatically reset mock state before every test
	// resetMocks: false,

	// Automatically restore mock state and implementation before every test
	// restoreMocks: false,

	// A preset that is used as a base for Jest's configuration
	preset: "ts-jest",

	// The test environment that will be used for testing
	// testEnvironment: "node",

	// A map from regular expressions to paths to transformers
	transform: {
		"^.+\\.tsx?$": [
			"ts-jest",
			{
				tsconfig: "tests/tsconfig.json",
				useESM: true,
			},
		],
	},

	transformIgnorePatterns: ["/node_modules/", String.raw`\.pnp\.[^\/]+$`],

	// An array of directory names to be searched recursively up from the requiring module's location
	moduleFileExtensions: ["js", "mjs", "cjs", "jsx", "ts", "tsx", "json", "node"],

	// Tells Jest to treat .ts/.tsx as ESM
	extensionsToTreatAsEsm: [".ts", ".tsx"],

	// Indicates whether each individual test should be reported during the run
	// verbose: undefined,

	// The paths to modules that run some code to configure or set up the testing environment before each test
	// setupFiles: [],
};

export default config;
