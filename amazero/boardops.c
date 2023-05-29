#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdint.h>

#define PIECE_NONE 0

typedef struct {
	int r;
	int c;
} vec2_t;

typedef struct {
	vec2_t queen_pick;
	vec2_t queen_move;
	vec2_t arrow_shot;
} move_t;

typedef struct {
	vec2_t dimensions;
	uint8_t* board;
} game_t;

static uint8_t get_piece(game_t game, vec2_t position) {
	return game.board[position.r * game.dimensions.c + position.c];
}

static size_t trace(game_t game, vec2_t position, vec2_t* out) {
	size_t i = 0;
	vec2_t dest;
	// up
	dest = position;
	while (--dest.r >= 0 && get_piece(game, dest) == PIECE_NONE) {
		out[i++] = dest;
	}
	// down
	dest = position;
	while (++dest.r < game.dimensions.r && get_piece(game, dest) == PIECE_NONE) {
		out[i++] = dest;
	}
	// left
	dest = position;
	while (--dest.c >= 0 && get_piece(game, dest) == PIECE_NONE) {
		out[i++] = dest;
	}
	// right
	dest = position;
	while (++dest.c < game.dimensions.c && get_piece(game, dest) == PIECE_NONE) {
		out[i++] = dest;
	}
	// up left
	dest = position;
	while (--dest.r >= 0 && --dest.c >= 0 && get_piece(game, dest) == PIECE_NONE) {
		out[i++] = dest;
	}
	// up right
	dest = position;
	while (--dest.r >= 0 && ++dest.c < game.dimensions.c && get_piece(game, dest) == PIECE_NONE) {
		out[i++] = dest;
	}
	// down left
	dest = position;
	while (++dest.r < game.dimensions.r && --dest.c >= 0 && get_piece(game, dest) == PIECE_NONE) {
		out[i++] = dest;
	}
	// down right
	dest = position;
	while (++dest.r < game.dimensions.r && ++dest.c < game.dimensions.c && get_piece(game, dest) == PIECE_NONE) {
		out[i++] = dest;
	}
	return i;
}

static size_t min_size(size_t a, size_t b) {
	if (a < b) {
		return a;
	} else {
		return b;
	}
}

static size_t max_trace_dim(vec2_t dim) {
	return dim.c-1 + dim.r-1 + 2 * min_size(dim.c-1, dim.r-1);
}

static size_t legal_moves(game_t game, vec2_t* queens, size_t queens_len, move_t* out) {
	size_t max_trace = max_trace_dim(game.dimensions);
	vec2_t queen_trace[max_trace], arrow_trace[max_trace];
	size_t queen_trace_len, arrow_trace_len;
	size_t i = 0;
	for (size_t q = 0; q < queens_len; q++) {
		queen_trace_len = trace(game, queens[q], queen_trace);
		uint8_t prev_piece = game.board[queens[q].r*game.dimensions.c+queens[q].c];
		game.board[queens[q].r*game.dimensions.c+queens[q].c] = PIECE_NONE;
		for (size_t m = 0; m < queen_trace_len; m++) {
			arrow_trace_len = trace(game, queen_trace[m], arrow_trace);
			for (size_t a = 0; a < arrow_trace_len; a++) {
				out[i].queen_pick = queens[q];
				out[i].queen_move = queen_trace[m];
				out[i].arrow_shot = arrow_trace[a];
				i += 1;
			}
		}
		game.board[queens[q].r*game.dimensions.c+queens[q].c] = prev_piece;
	}
	return i;
}

static PyObject *boardops_legal_moves(PyObject *self, PyObject *args) {
	PyArrayObject *boardarr=NULL, *queensarr=NULL;
	game_t game;
	vec2_t* queens;
	int queens_len;

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &boardarr, &PyArray_Type, &queensarr)) return NULL;

	if (!PyArray_ISCARRAY(boardarr)) { PyErr_SetString(PyExc_ValueError, "board must be a c-array"); return NULL; }
	if (PyArray_NDIM(boardarr) != 2) { PyErr_SetString(PyExc_ValueError, "board must 2-dimensional"); return NULL; }
	if (PyArray_TYPE(boardarr) != NPY_UBYTE) { PyErr_SetString(PyExc_ValueError, "board must be type ubyte"); return NULL; }
	game.board = PyArray_DATA(boardarr);
	game.dimensions.r = PyArray_SHAPE(boardarr)[0];
	game.dimensions.c = PyArray_SHAPE(boardarr)[1];
	
	if (!PyArray_ISCARRAY(queensarr)) { PyErr_SetString(PyExc_ValueError, "queens must be a c-array"); return NULL; }
	if (PyArray_NDIM(queensarr) != 2) { PyErr_SetString(PyExc_ValueError, "queens must 2-dimensional"); return NULL; }
	if (PyArray_SHAPE(queensarr)[1] != 2) { PyErr_SetString(PyExc_ValueError, "queens dim 1 must be 2"); return NULL; }
	if (PyArray_TYPE(queensarr) != NPY_INT) { PyErr_SetString(PyExc_ValueError, "queens must be type int"); return NULL; }
	queens = PyArray_DATA(queensarr);
	queens_len = PyArray_SHAPE(queensarr)[0];

	size_t max_trace = max_trace_dim(game.dimensions);
	move_t* out = malloc(queens_len*max_trace*max_trace*sizeof(move_t));
	if (!out) return PyErr_NoMemory();

	size_t move_len = legal_moves(game, queens, queens_len, out);

	out = realloc(out, move_len*sizeof(move_t));
	if (!out) return PyErr_NoMemory();
	
	npy_intp dims[3] = {move_len,3,2};
	PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNewFromData(3, dims, NPY_INT, out);
	PyArray_ENABLEFLAGS(result, NPY_ARRAY_OWNDATA);

	return result;
}

static PyMethodDef BoardopsMethods[] = {
    {"legal_moves",  boardops_legal_moves, METH_VARARGS, "Determine legal moves in the game of amazons."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef boardopsmodule = {
    PyModuleDef_HEAD_INIT,
    "boardops",
    NULL,
    -1,
    BoardopsMethods
};

PyMODINIT_FUNC
PyInit_boardops(void)
{
	import_array();
    return PyModule_Create(&boardopsmodule);
}
