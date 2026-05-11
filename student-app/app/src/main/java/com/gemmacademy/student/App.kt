package com.gemmacademy.student

import android.app.Application
import com.gemmacademy.student.model.ModelInference

class App : Application() {
    val modelInference: ModelInference by lazy { ModelInference(applicationContext) }
}
