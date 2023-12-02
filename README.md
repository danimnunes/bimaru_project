# Project: Bimaru Solver

## Introduction

This Python program is designed to solve the Bimaru problem, also known as the Battleship Puzzle or Solitaire Battleship, using AI techniques.

## Problem Description

The Bimaru game is played on a 10x10 grid representing an ocean area. The player's objective is to find a hidden fleet consisting of a battleship (four squares), two cruisers (three squares each), three destroyers (two squares each), and four submarines (one square each). Ships can be oriented horizontally or vertically, and no two ships can occupy adjacent grid squares, even diagonally.

The player receives row and column counts indicating the number of occupied squares in each row and column, along with various hints specifying the state of individual squares on the grid (water, circle, middle, top, bottom, left, or right).

## Objective

The goal of this project is to develop a Python program (`bimaru.py`) that takes a Bimaru instance as input and returns a fully filled grid as output. The program must read the Bimaru instance from standard input and solve the problem using AI techniques. The solution should be printed to standard output.

## Usage

```bash
python3 bimaru.py < <instance_file>

