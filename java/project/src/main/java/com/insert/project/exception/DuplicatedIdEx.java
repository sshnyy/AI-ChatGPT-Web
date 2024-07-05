package com.insert.project.exception;

public class DuplicatedIdEx extends RuntimeException{
    public DuplicatedIdEx() {
        super();
    }

    public DuplicatedIdEx(String message) {
        super(message);
    }

    public DuplicatedIdEx(String message, Throwable cause) {
        super(message, cause);
    }

    public DuplicatedIdEx(Throwable cause) {
        super(cause);
    }

    protected DuplicatedIdEx(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
