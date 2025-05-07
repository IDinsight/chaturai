/**
 * @file This file contains the ESLint configuration for the project.
 */
/* eslint-disable import/no-unresolved */
// eslint-disable-next-line unicorn/import-style,import/no-nodejs-modules
import {dirname} from "node:path";
// eslint-disable-next-line import/no-nodejs-modules
import {fileURLToPath} from "node:url";

import {FlatCompat} from "@eslint/eslintrc";
import js from "@eslint/js";
import ts from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import pluginImport from "eslint-plugin-import";
import pluginJest from "eslint-plugin-jest";
import pluginJSDoc from "eslint-plugin-jsdoc";
import pluginReact from "eslint-plugin-react";
import pluginUnicorn from "eslint-plugin-unicorn";
import globals from "globals";
/* eslint-enable import/no-unresolved */

// Detect if current working directory ends with "frontend". This is required to point
// to the correct tsconfig.json file depending on where the eslint command is being
// executed from.
const cwd = process.cwd();
const isInFrontend = cwd.endsWith("frontend");
const tsconfigPath = isInFrontend ? "./tsconfig.json" : "./frontend/tsconfig.json";

// Get the current directory for FlatCompat.
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
	baseDirectory: __dirname,
});

/**
 * Common rules for both JS & TS files (no TS-specific checks).
 * TSDoc/TS-only rules are applied in the typed-linting block below.
 */
const baseRules = {
	// JavaScript core recommended.
	...js.configs.recommended.rules,

	// ESLint plugin import recommended.
	...pluginImport.configs.recommended.rules,

	// Jest recommended.
	...pluginJest.configs.recommended.rules,

	// JSDoc recommended.
	...pluginJSDoc.configs.recommended.rules,

	// React flat recommended.
	...pluginReact.configs.flat.recommended.rules,

	// Unicorn recommended rules.
	...pluginUnicorn.configs.recommended.rules,

	/**
	 * Custom JSDoc rules. Will also apply to TS files.
	 * NB: It is recommended to NOT use TSDoc in combination with JSDoc since TSDoc has
	 * a different syntax and is sometimes not compatible with JSDoc. Pick one or the
	 * other.
	 */
	"jsdoc/check-values": "warn",
	"jsdoc/informative-docs": "warn",
	"jsdoc/require-asterisk-prefix": "warn",
	"jsdoc/require-description": "warn",
	"jsdoc/require-description-complete-sentence": "warn",
	"jsdoc/require-file-overview": "error",
	"jsdoc/require-hyphen-before-param-description": "error",
	"jsdoc/require-jsdoc": "warn",
	"jsdoc/require-param": "warn",
	"jsdoc/require-param-type": "off",
	"jsdoc/require-property-name": "warn",
	"jsdoc/require-property-type": "off",
	"jsdoc/require-returns": "warn",
	"jsdoc/require-returns-type": "off",
	"jsdoc/require-throws": "warn",
	"jsdoc/require-yields": "warn",
	"jsdoc/require-yields-check": "warn",
	"jsdoc/sort-tags": "off",
	"jsdoc/tag-lines": "off",

	/**
	 * Custom Import plugin rules (enforce ordering, prevent duplicates, etc.).
	 */
	"import/default": "error",
	"import/first": "error",
	"import/named": "error",
	"import/newline-after-import": "error",
	"import/no-absolute-path": "error",
	"import/no-cycle": "error",
	"import/no-deprecated": "warn",
	"import/no-duplicates": "error",
	"import/no-dynamic-require": "warn",
	"import/no-extraneous-dependencies": "error",
	"import/no-named-as-default": "error",
	"import/no-nodejs-modules": "warn",
	"import/no-self-import": "error",
	"import/no-unresolved": "error",
	"import/order": [
		"error",
		{
			groups: ["builtin", "external", "internal", "parent", "sibling", "index"],
			"newlines-between": "always",
			alphabetize: {order: "asc", caseInsensitive: true},
			pathGroups: [
				// Treat React specially and put it at the top of 'external'
				{
					group: "external",
					pattern: "react",
					position: "before",
				},
			],
			pathGroupsExcludedImportTypes: ["builtin"],
		},
	],

	/**
	 * Custom Unicorn plugin rules.
	 */
	"unicorn/better-regex": "warn",
	"unicorn/catch-error-name": "warn",
	"unicorn/empty-brace-spaces": "warn",
	"unicorn/no-for-loop": "error",
	"unicorn/no-useless-length-check": "error",
	"unicorn/prefer-switch": "warn",
	"unicorn/prefer-top-level-await": "warn",
	"unicorn/prefer-ternary": "warn",
};

/**
 * Extra rules specifically for typed-linting (TypeScript + TSDoc).
 */
const typedRules = {
	// TypeScript ESLint recommended.
	...ts.configs.recommended.rules,
};

export default [
	// Global ignore patterns.
	{
		ignores: [
			".stylelintrc.js",
			"**/build/**",
			"**/coverage/**",
			"**/dist/**",
			"**/node_modules/**",
		],
	},

	// Next.js and TypeScript configuration via FlatCompat.
	...compat.extends("next/core-web-vitals", "next/typescript"),

	// TS typed-linting (files: *.ts, *.tsx).
	{
		files: ["**/*.ts", "**/*.tsx"],
		languageOptions: {
			parser: tsParser,
			parserOptions: {
				ecmaVersion: "latest",
				sourceType: "module",
				tsconfigRootDir: cwd,
				project: tsconfigPath, // typed-linting requires this
			},
			globals: {
				...globals.browser,
				...globals.jest,
				...globals.node,
			},
		},
		plugins: {
			"@typescript-eslint": ts,
			import: pluginImport,
			jest: pluginJest,
			jsdoc: pluginJSDoc,
			react: pluginReact,
			unicorn: pluginUnicorn,
		},
		rules: {
			...baseRules,
			...typedRules,
		},
		settings: {
			"import/resolver": {
				typescript: {
					project: tsconfigPath,
				},
			},
			react: {
				version: "detect", // Automatically detect the version of React
			},
		},
	},

	// JS linting (files: *.js, *.jsx, *.cjs, *.mjs).
	{
		files: ["**/*.js", "**/*.jsx", "**/*.cjs", "**/*.mjs"],
		languageOptions: {
			/**
			 * We still can use `tsParser` to parse JS (it can parse JS syntax fine),
			 * but we do NOT provide `project`, so there's no typed-linting.
			 */
			parser: tsParser,
			parserOptions: {
				ecmaVersion: "latest",
				sourceType: "module",
			},
			globals: {
				...globals.browser,
				...globals.jest,
				...globals.node,
			},
		},
		plugins: {
			"@typescript-eslint": ts,
			import: pluginImport,
			jest: pluginJest,
			jsdoc: pluginJSDoc,
			react: pluginReact,
			unicorn: pluginUnicorn,
		},
		rules: {
			...baseRules,
		},
	},
];
