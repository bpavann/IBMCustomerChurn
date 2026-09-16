pipeline {
agent any

```
stages {

    stage('Checkout') {
        steps {
            echo 'Checking out source code...'
            checkout scm
        }
    }

    stage('Install Dependencies') {
        steps {
            echo 'Installing Python dependencies...'
            sh 'pip install -r requirements.txt'
        }
    }

    stage('Build Docker Image') {
        steps {
            echo 'Building Docker image...'
            sh 'docker build -t customer-churn-ibm .'
        }
    }

    stage('Docker Build Complete') {
        steps {
            echo 'Customer Churn Docker image built successfully.'
        }
    }
}

post {
    success {
        echo 'CI pipeline completed successfully.'
    }

    failure {
        echo 'CI pipeline failed.'
    }
}
```

}
