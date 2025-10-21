from PyQt5 import QtCore, QtWidgets, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 700)


        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 700))


        self.centralWidget = QtWidgets.QWidget(parent=MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.centralWidget.setObjectName("centralWidget")


        self.layoutCentral = QtWidgets.QHBoxLayout(self.centralWidget)
        self.layoutCentral.setContentsMargins(4, 4, 4, 4)
        self.layoutCentral.setSpacing(4)
        self.layoutCentral.setObjectName("layoutCentral")


        self.computePanel = QtWidgets.QWidget(parent=self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.computePanel.sizePolicy().hasHeightForWidth())
        self.computePanel.setSizePolicy(sizePolicy)
        self.computePanel.setMinimumSize(QtCore.QSize(0, 0))
        self.computePanel.setMaximumSize(QtCore.QSize(300, 16777215))
        self.computePanel.setBaseSize(QtCore.QSize(0, 0))
        self.computePanel.setStyleSheet("")
        self.computePanel.setObjectName("computePanel")


        self.layoutCompute = QtWidgets.QVBoxLayout(self.computePanel)
        self.layoutCompute.setContentsMargins(4, 4, 4, 4)
        self.layoutCompute.setSpacing(4)
        self.layoutCompute.setObjectName("layoutCompute")


        self.titlePanel = QtWidgets.QFrame(parent=self.computePanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.titlePanel.sizePolicy().hasHeightForWidth())
        self.titlePanel.setSizePolicy(sizePolicy)
        self.titlePanel.setMaximumSize(QtCore.QSize(16777215, 60))
        self.titlePanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.titlePanel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.titlePanel.setObjectName("titlePanel")


        self.layoutTitlePanel = QtWidgets.QVBoxLayout(self.titlePanel)
        self.layoutTitlePanel.setSpacing(4)
        self.layoutTitlePanel.setObjectName("layoutTitlePanel")


        self.labelTitle = QtWidgets.QLabel(parent=self.titlePanel)
        self.labelTitle.setStyleSheet("")
        self.labelTitle.setScaledContents(False)
        self.labelTitle.setObjectName("labelTitle")


        self.layoutTitlePanel.addWidget(self.labelTitle)
        self.layoutCompute.addWidget(self.titlePanel)


        self.dataPanel = QtWidgets.QFrame(parent=self.computePanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dataPanel.sizePolicy().hasHeightForWidth())
        self.dataPanel.setSizePolicy(sizePolicy)
        self.dataPanel.setMinimumSize(QtCore.QSize(0, 0))
        self.dataPanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.dataPanel.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.dataPanel.setLineWidth(1)
        self.dataPanel.setMidLineWidth(0)
        self.dataPanel.setObjectName("dataPanel")


        self.layoutDataPanel = QtWidgets.QVBoxLayout(self.dataPanel)
        self.layoutDataPanel.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint)
        self.layoutDataPanel.setContentsMargins(9, 9, 9, 9)
        self.layoutDataPanel.setSpacing(4)
        self.layoutDataPanel.setObjectName("layoutDataPanel")


        self.labelDataSelect = QtWidgets.QLabel(parent=self.dataPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelDataSelect.sizePolicy().hasHeightForWidth())
        self.labelDataSelect.setSizePolicy(sizePolicy)
        self.labelDataSelect.setMaximumSize(QtCore.QSize(16777215, 20))
        self.labelDataSelect.setStyleSheet("font: 9pt \"Sans Serif\";\n"
"text-decoration: underline;")
        self.labelDataSelect.setObjectName("labelDataSelect")


        self.layoutDataPanel.addWidget(self.labelDataSelect)


        self.selectDataCase = QtWidgets.QComboBox(parent=self.dataPanel)
        self.selectDataCase.setStyleSheet("")
        self.selectDataCase.setObjectName("selectDataCase")
        self.selectDataCase.addItem("")
        self.selectDataCase.addItem("")
        self.layoutDataPanel.addWidget(self.selectDataCase)


        self.dataSubPanel = QtWidgets.QWidget(parent=self.dataPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dataSubPanel.sizePolicy().hasHeightForWidth())
        self.dataSubPanel.setSizePolicy(sizePolicy)
        self.dataSubPanel.setObjectName("dataSubPanel")


        self.layoutDataGrid = QtWidgets.QGridLayout(self.dataSubPanel)
        self.layoutDataGrid.setContentsMargins(4, 4, 4, 4)
        self.layoutDataGrid.setSpacing(4)
        self.layoutDataGrid.setObjectName("layoutDataGrid")


        self.pushCustomCase = QtWidgets.QPushButton(parent=self.dataSubPanel)
        self.pushCustomCase.setObjectName("pushCustomCase")


        self.layoutDataGrid.addWidget(self.pushCustomCase, 0, 0, 1, 1)
        self.pushImportData = QtWidgets.QPushButton(parent=self.dataSubPanel)
        self.pushImportData.setObjectName("pushImportData")
        self.layoutDataGrid.addWidget(self.pushImportData, 0, 1, 1, 1)


        self.lineEdCV = QtWidgets.QLineEdit(parent=self.dataSubPanel)
        self.lineEdCV.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdCV.sizePolicy().hasHeightForWidth())
        self.lineEdCV.setSizePolicy(sizePolicy)
        self.lineEdCV.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.lineEdCV.setObjectName("lineEdCV")
        self.layoutDataGrid.addWidget(self.lineEdCV, 1, 1, 1, 1)


        self.lineEdDataSize = QtWidgets.QLineEdit(parent=self.dataSubPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdDataSize.sizePolicy().hasHeightForWidth())
        self.lineEdDataSize.setSizePolicy(sizePolicy)
        self.lineEdDataSize.setInputMask("")
        self.lineEdDataSize.setText("")
        self.lineEdDataSize.setFrame(True)
        self.lineEdDataSize.setClearButtonEnabled(False)
        self.lineEdDataSize.setObjectName("lineEdDataSize")
        self.layoutDataGrid.addWidget(self.lineEdDataSize, 1, 0, 1, 1)


        self.layoutDataPanel.addWidget(self.dataSubPanel)
        self.layoutCompute.addWidget(self.dataPanel)


        self.regModelPanel = QtWidgets.QFrame(parent=self.computePanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.regModelPanel.sizePolicy().hasHeightForWidth())
        self.regModelPanel.setSizePolicy(sizePolicy)
        self.regModelPanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.regModelPanel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.regModelPanel.setObjectName("regModelPanel")


        self.layoutRegModelPanel = QtWidgets.QVBoxLayout(self.regModelPanel)
        self.layoutRegModelPanel.setContentsMargins(9, 9, 9, 9)
        self.layoutRegModelPanel.setSpacing(4)
        self.layoutRegModelPanel.setObjectName("layoutRegModelPanel")


        self.labelRegModel = QtWidgets.QLabel(parent=self.regModelPanel)
        self.labelRegModel.setMaximumSize(QtCore.QSize(16777215, 20))
        self.labelRegModel.setStyleSheet("font: 9pt \"Sans Serif\";\n"
"text-decoration: underline;")
        self.labelRegModel.setObjectName("labelRegModel")


        self.layoutRegModelPanel.addWidget(self.labelRegModel)


        self.selectRegMod = QtWidgets.QComboBox(parent=self.regModelPanel)
        self.selectRegMod.setStyleSheet("")
        self.selectRegMod.setObjectName("selectRegMod")
        self.selectRegMod.addItem("")
        self.selectRegMod.addItem("")
        self.selectRegMod.addItem("")
        self.selectRegMod.addItem("")


        self.layoutRegModelPanel.addWidget(self.selectRegMod)


        self.regModelSubPanel = QtWidgets.QWidget(parent=self.regModelPanel)
        self.regModelSubPanel.setObjectName("regModelSubPanel")


        self.layoutRegModelSubPanel = QtWidgets.QHBoxLayout(self.regModelSubPanel)
        self.layoutRegModelSubPanel.setContentsMargins(4, 4, 4, 4)
        self.layoutRegModelSubPanel.setSpacing(4)
        self.layoutRegModelSubPanel.setObjectName("layoutRegModelSubPanel")


        self.pushParamReg = QtWidgets.QPushButton(parent=self.regModelSubPanel)
        self.pushParamReg.setObjectName("pushParamReg")


        self.layoutRegModelSubPanel.addWidget(self.pushParamReg)


        self.pushFitReg = QtWidgets.QPushButton(parent=self.regModelSubPanel)
        self.pushFitReg.setObjectName("pushFitReg")


        self.layoutRegModelSubPanel.addWidget(self.pushFitReg)
        self.layoutRegModelPanel.addWidget(self.regModelSubPanel)
        self.layoutCompute.addWidget(self.regModelPanel)


        self.bayesModelPanel = QtWidgets.QFrame(parent=self.computePanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bayesModelPanel.sizePolicy().hasHeightForWidth())
        self.bayesModelPanel.setSizePolicy(sizePolicy)
        self.bayesModelPanel.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.bayesModelPanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.bayesModelPanel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.bayesModelPanel.setObjectName("bayesModelPanel")


        self.layoutBayesModel = QtWidgets.QVBoxLayout(self.bayesModelPanel)
        self.layoutBayesModel.setContentsMargins(9, 9, 9, 9)
        self.layoutBayesModel.setSpacing(4)
        self.layoutBayesModel.setObjectName("layoutBayesModel")


        self.labelBayesModel = QtWidgets.QLabel(parent=self.bayesModelPanel)
        self.labelBayesModel.setMaximumSize(QtCore.QSize(16777215, 20))
        self.labelBayesModel.setStyleSheet("font: 9pt \"Sans Serif\";\n"
"text-decoration: underline;")
        self.labelBayesModel.setObjectName("labelBayesModel")


        self.layoutBayesModel.addWidget(self.labelBayesModel)


        self.selectBayesModel = QtWidgets.QComboBox(parent=self.bayesModelPanel)
        self.selectBayesModel.setStyleSheet("")
        self.selectBayesModel.setObjectName("selectBayesModel")
        self.selectBayesModel.addItem("")
        self.selectBayesModel.addItem("")
        self.selectBayesModel.addItem("")


        self.layoutBayesModel.addWidget(self.selectBayesModel)


        self.bayesModelSubPanel = QtWidgets.QWidget(parent=self.bayesModelPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bayesModelSubPanel.sizePolicy().hasHeightForWidth())
        self.bayesModelSubPanel.setSizePolicy(sizePolicy)
        self.bayesModelSubPanel.setObjectName("bayesModelSubPanel")


        self.layoutBayesModelSubPanel = QtWidgets.QHBoxLayout(self.bayesModelSubPanel)
        self.layoutBayesModelSubPanel.setContentsMargins(4, 4, 4, 4)
        self.layoutBayesModelSubPanel.setSpacing(4)
        self.layoutBayesModelSubPanel.setObjectName("layoutBayesModelSubPanel")


        self.selectParamTune = QtWidgets.QComboBox(parent=self.bayesModelSubPanel)
        self.selectParamTune.setObjectName("selectParamTune")
        self.selectParamTune.addItem("")


        self.layoutBayesModelSubPanel.addWidget(self.selectParamTune)


        self.bayesTuningPanel = QtWidgets.QWidget(parent=self.bayesModelSubPanel)
        self.bayesTuningPanel.setObjectName("bayesTuningPanel")


        self.layoutBayesGrid = QtWidgets.QGridLayout(self.bayesTuningPanel)
        self.layoutBayesGrid.setContentsMargins(4, 4, 4, 4)
        self.layoutBayesGrid.setSpacing(4)
        self.layoutBayesGrid.setObjectName("layoutBayesGrid")


        self.selectDistType = QtWidgets.QComboBox(parent=self.bayesTuningPanel)
        self.selectDistType.setObjectName("selectDistType")
        self.selectDistType.addItem("")
        self.selectDistType.addItem("")
        self.selectDistType.addItem("")
        self.selectDistType.addItem("")


        self.layoutBayesGrid.addWidget(self.selectDistType, 0, 0, 1, 1)


        self.pushParamAuto = QtWidgets.QPushButton(parent=self.bayesTuningPanel)
        self.pushParamAuto.setObjectName("pushParamAuto")


        self.layoutBayesGrid.addWidget(self.pushParamAuto, 0, 1, 1, 1)


        self.lineEdpar1 = QtWidgets.QLineEdit(parent=self.bayesTuningPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdpar1.sizePolicy().hasHeightForWidth())
        self.lineEdpar1.setSizePolicy(sizePolicy)
        self.lineEdpar1.setObjectName("lineEdpar1")


        self.layoutBayesGrid.addWidget(self.lineEdpar1, 1, 0, 1, 1)


        self.lineEdpar2 = QtWidgets.QLineEdit(parent=self.bayesTuningPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdpar2.sizePolicy().hasHeightForWidth())
        self.lineEdpar2.setSizePolicy(sizePolicy)
        self.lineEdpar2.setObjectName("lineEdpar2")


        self.layoutBayesGrid.addWidget(self.lineEdpar2, 1, 1, 1, 1)
        self.layoutBayesModelSubPanel.addWidget(self.bayesTuningPanel)
        self.layoutBayesModel.addWidget(self.bayesModelSubPanel)
        self.layoutCompute.addWidget(self.bayesModelPanel)
        self.inferencePanel = QtWidgets.QFrame(parent=self.computePanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inferencePanel.sizePolicy().hasHeightForWidth())
        self.inferencePanel.setSizePolicy(sizePolicy)
        self.inferencePanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.inferencePanel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.inferencePanel.setObjectName("inferencePanel")


        self.layoutInferencePanel = QtWidgets.QVBoxLayout(self.inferencePanel)
        self.layoutInferencePanel.setContentsMargins(9, 9, 9, 9)
        self.layoutInferencePanel.setSpacing(4)
        self.layoutInferencePanel.setObjectName("layoutInferencePanel")


        self.labelInferenceConfig = QtWidgets.QLabel(parent=self.inferencePanel)
        self.labelInferenceConfig.setMaximumSize(QtCore.QSize(16777215, 20))
        self.labelInferenceConfig.setStyleSheet("font: 9pt \"Sans Serif\";\n"
"text-decoration: underline;")
        self.labelInferenceConfig.setObjectName("labelInferenceConfig")


        self.layoutInferencePanel.addWidget(self.labelInferenceConfig)


        self.inferenceSubPanel = QtWidgets.QWidget(parent=self.inferencePanel)
        self.inferenceSubPanel.setObjectName("inferenceSubPanel")


        self.layoutInferenceSubPanel = QtWidgets.QGridLayout(self.inferenceSubPanel)
        self.layoutInferenceSubPanel.setContentsMargins(4, 4, 4, 4)
        self.layoutInferenceSubPanel.setSpacing(4)
        self.layoutInferenceSubPanel.setObjectName("layoutInferenceSubPanel")


        self.lineEdMC1 = QtWidgets.QLineEdit(parent=self.inferenceSubPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdMC1.sizePolicy().hasHeightForWidth())
        self.lineEdMC1.setSizePolicy(sizePolicy)
        self.lineEdMC1.setMaximumSize(QtCore.QSize(80, 16777215))
        self.lineEdMC1.setObjectName("lineEdMC1")


        self.layoutInferenceSubPanel.addWidget(self.lineEdMC1, 1, 0, 1, 1)


        self.lineEdMC3 = QtWidgets.QLineEdit(parent=self.inferenceSubPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdMC3.sizePolicy().hasHeightForWidth())
        self.lineEdMC3.setSizePolicy(sizePolicy)
        self.lineEdMC3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.lineEdMC3.setObjectName("lineEdMC3")


        self.layoutInferenceSubPanel.addWidget(self.lineEdMC3, 1, 2, 1, 1)
        self.radioStepType1 = QtWidgets.QRadioButton(parent=self.inferenceSubPanel)
        self.radioStepType1.setMaximumSize(QtCore.QSize(80, 16777215))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.radioStepType1.setFont(font)
        self.radioStepType1.setObjectName("radioStepType1")


        self.layoutInferenceSubPanel.addWidget(self.radioStepType1, 0, 0, 1, 1)


        self.radioStepType2 = QtWidgets.QRadioButton(parent=self.inferenceSubPanel)
        self.radioStepType2.setMaximumSize(QtCore.QSize(80, 16777215))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.radioStepType2.setFont(font)
        self.radioStepType2.setObjectName("radioStepType2")


        self.layoutInferenceSubPanel.addWidget(self.radioStepType2, 0, 1, 1, 1)


        self.lineEdMC2 = QtWidgets.QLineEdit(parent=self.inferenceSubPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdMC2.sizePolicy().hasHeightForWidth())
        self.lineEdMC2.setSizePolicy(sizePolicy)
        self.lineEdMC2.setMaximumSize(QtCore.QSize(80, 16777215))
        self.lineEdMC2.setObjectName("lineEdMC2")


        self.layoutInferenceSubPanel.addWidget(self.lineEdMC2, 1, 1, 1, 1)
        self.layoutInferencePanel.addWidget(self.inferenceSubPanel)
        self.computeSubPanel = QtWidgets.QWidget(parent=self.inferencePanel)
        self.computeSubPanel.setObjectName("computeSubPanel")


        self.layoutComputeSubPanel = QtWidgets.QHBoxLayout(self.computeSubPanel)
        self.layoutComputeSubPanel.setObjectName("layoutComputeSubPanel")


        self.pushCompute = QtWidgets.QPushButton(parent=self.computeSubPanel)
        self.pushCompute.setObjectName("pushCompute")


        self.layoutComputeSubPanel.addWidget(self.pushCompute)
        self.pushExport = QtWidgets.QPushButton(parent=self.computeSubPanel)
        self.pushExport.setObjectName("pushExport")


        self.layoutComputeSubPanel.addWidget(self.pushExport)
        self.layoutInferencePanel.addWidget(self.computeSubPanel)
        self.layoutCompute.addWidget(self.inferencePanel)


        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.layoutCompute.addItem(spacerItem)


        self.layoutCentral.addWidget(self.computePanel)


        self.displayPanel = QtWidgets.QWidget(parent=self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.displayPanel.sizePolicy().hasHeightForWidth())
        self.displayPanel.setSizePolicy(sizePolicy)
        self.displayPanel.setMinimumSize(QtCore.QSize(300, 0))
        self.displayPanel.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.displayPanel.setStyleSheet("")
        self.displayPanel.setObjectName("displayPanel")


        self.layoutPanel = QtWidgets.QVBoxLayout(self.displayPanel)
        self.layoutPanel.setContentsMargins(4, 4, 4, 4)
        self.layoutPanel.setSpacing(4)
        self.layoutPanel.setObjectName("layoutPanel")


        self.tabViewPanel = QtWidgets.QTabWidget(parent=self.displayPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabViewPanel.sizePolicy().hasHeightForWidth())
        self.tabViewPanel.setSizePolicy(sizePolicy)
        self.tabViewPanel.setMinimumSize(QtCore.QSize(0, 0))
        self.tabViewPanel.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.tabViewPanel.setStyleSheet("")
        self.tabViewPanel.setObjectName("tabViewPanel")


        self.tab = QtWidgets.QWidget()
        self.tab.setStyleSheet("")
        self.tab.setObjectName("tab")


        self.tabViewPanel.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setStyleSheet("")
        self.tab_2.setObjectName("tab_2")


        self.tabViewPanel.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setStyleSheet("")
        self.tab_3.setObjectName("tab_3")


        self.tabViewPanel.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setStyleSheet("")
        self.tab_4.setObjectName("tab_4")


        self.tabViewPanel.addTab(self.tab_4, "")
        self.layoutPanel.addWidget(self.tabViewPanel)
        self.subDisplayPanel = QtWidgets.QFrame(parent=self.displayPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.subDisplayPanel.sizePolicy().hasHeightForWidth())
        self.subDisplayPanel.setSizePolicy(sizePolicy)
        self.subDisplayPanel.setMinimumSize(QtCore.QSize(0, 220))
        self.subDisplayPanel.setMaximumSize(QtCore.QSize(16777215, 300))
        self.subDisplayPanel.setBaseSize(QtCore.QSize(0, 0))
        self.subDisplayPanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.subDisplayPanel.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.subDisplayPanel.setObjectName("subDisplayPanel")


        self.layoutDisplaySubPanel = QtWidgets.QHBoxLayout(self.subDisplayPanel)
        self.layoutDisplaySubPanel.setObjectName("layoutDisplaySubPanel")


        self.displayControl = QtWidgets.QFrame(parent=self.subDisplayPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.displayControl.sizePolicy().hasHeightForWidth())
        self.displayControl.setSizePolicy(sizePolicy)
        self.displayControl.setMinimumSize(QtCore.QSize(360, 0))
        self.displayControl.setMaximumSize(QtCore.QSize(320, 16777215))
        self.displayControl.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.displayControl.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.displayControl.setObjectName("displayControl")


        self.layoutDisplayControl = QtWidgets.QVBoxLayout(self.displayControl)
        self.layoutDisplayControl.setObjectName("layoutDisplayControl")


        self.labelDisplay = QtWidgets.QLabel(parent=self.displayControl)
        self.labelDisplay.setMaximumSize(QtCore.QSize(16777215, 20))
        self.labelDisplay.setStyleSheet("text-decoration: underline;\n"
"font: 9pt \"Sans Serif\";")
        self.labelDisplay.setObjectName("labelDisplay")


        self.layoutDisplayControl.addWidget(self.labelDisplay)
        self.displayFineControl = QtWidgets.QWidget(parent=self.displayControl)
        self.displayFineControl.setObjectName("displayFineControl")


        self.layoutDisplayFineControl = QtWidgets.QGridLayout(self.displayFineControl)
        self.layoutDisplayFineControl.setContentsMargins(5, 5, 5, 5)
        self.layoutDisplayFineControl.setSpacing(2)
        self.layoutDisplayFineControl.setObjectName("layoutDisplayFineControl")


        self.legendSelectPanel = QtWidgets.QFrame(parent=self.displayFineControl)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.legendSelectPanel.sizePolicy().hasHeightForWidth())
        self.legendSelectPanel.setSizePolicy(sizePolicy)
        self.legendSelectPanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.legendSelectPanel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.legendSelectPanel.setObjectName("legendSelectPanel")


        self.layoutControlVert1 = QtWidgets.QVBoxLayout(self.legendSelectPanel)
        self.layoutControlVert1.setObjectName("layoutControlVert1")


        self.checkLegend1 = QtWidgets.QCheckBox(parent=self.legendSelectPanel)
        self.checkLegend1.setObjectName("checkLegend1")
        self.checkLegend1.setChecked(True)


        self.layoutControlVert1.addWidget(self.checkLegend1)
        self.checkLegend2 = QtWidgets.QCheckBox(parent=self.legendSelectPanel)
        self.checkLegend2.setObjectName("checkLegend2")
        self.checkLegend2.setChecked(True)


        self.layoutControlVert1.addWidget(self.checkLegend2)
        self.checkLegend3 = QtWidgets.QCheckBox(parent=self.legendSelectPanel)
        self.checkLegend3.setObjectName("checkLegend3")
        self.checkLegend3.setChecked(True)

        self.layoutControlVert1.addWidget(self.checkLegend3)
        self.checkLegend4 = QtWidgets.QCheckBox(parent=self.legendSelectPanel)
        self.checkLegend4.setObjectName("checkLegend4")
        self.checkLegend4.setChecked(True)

        self.layoutControlVert1.addWidget(self.checkLegend4)
        self.checkLegend5 = QtWidgets.QCheckBox(parent=self.legendSelectPanel)
        self.checkLegend5.setObjectName("checkLegend5")
        self.checkLegend5.setChecked(True)

        self.layoutControlVert1.addWidget(self.checkLegend5)
        self.layoutDisplayFineControl.addWidget(self.legendSelectPanel, 1, 0, 1, 1)
        self.rangeSelectionPanel = QtWidgets.QFrame(parent=self.displayFineControl)
        self.rangeSelectionPanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.rangeSelectionPanel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.rangeSelectionPanel.setObjectName("rangeSelectionPanel")


        self.layoutControlGrid2 = QtWidgets.QGridLayout(self.rangeSelectionPanel)
        self.layoutControlGrid2.setObjectName("layoutControlGrid2")


        self.lineEdXmax = QtWidgets.QLineEdit(parent=self.rangeSelectionPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdXmax.sizePolicy().hasHeightForWidth())
        self.lineEdXmax.setSizePolicy(sizePolicy)
        self.lineEdXmax.setMaximumSize(QtCore.QSize(80, 16777215))
        self.lineEdXmax.setObjectName("lineEdXmax")


        self.layoutControlGrid2.addWidget(self.lineEdXmax, 1, 1, 1, 1)
        self.pushAutoRange = QtWidgets.QPushButton(parent=self.rangeSelectionPanel)
        self.pushAutoRange.setObjectName("pushAutoRange")


        self.layoutControlGrid2.addWidget(self.pushAutoRange, 0, 1, 1, 1)
        self.labelRange = QtWidgets.QLabel(parent=self.rangeSelectionPanel)
        self.labelRange.setObjectName("labelRange")


        self.layoutControlGrid2.addWidget(self.labelRange, 0, 0, 1, 1)
        self.lineEdXmin = QtWidgets.QLineEdit(parent=self.rangeSelectionPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdXmin.sizePolicy().hasHeightForWidth())
        self.lineEdXmin.setSizePolicy(sizePolicy)
        self.lineEdXmin.setMaximumSize(QtCore.QSize(80, 16777215))
        self.lineEdXmin.setObjectName("lineEdXmin")


        self.layoutControlGrid2.addWidget(self.lineEdXmin, 1, 0, 1, 1)
        self.lineEdYmin = QtWidgets.QLineEdit(parent=self.rangeSelectionPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdYmin.sizePolicy().hasHeightForWidth())
        self.lineEdYmin.setSizePolicy(sizePolicy)
        self.lineEdYmin.setMaximumSize(QtCore.QSize(80, 16777215))
        self.lineEdYmin.setObjectName("lineEdYmin")


        self.layoutControlGrid2.addWidget(self.lineEdYmin, 2, 0, 1, 1)
        self.lineEdYmax = QtWidgets.QLineEdit(parent=self.rangeSelectionPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdYmax.sizePolicy().hasHeightForWidth())
        self.lineEdYmax.setSizePolicy(sizePolicy)
        self.lineEdYmax.setMaximumSize(QtCore.QSize(80, 16777215))
        self.lineEdYmax.setObjectName("lineEdYmax")


        self.layoutControlGrid2.addWidget(self.lineEdYmax, 2, 1, 1, 1)
        self.layoutDisplayFineControl.addWidget(self.rangeSelectionPanel, 0, 1, 1, 1)
        self.subVisuPanel = QtWidgets.QFrame(parent=self.displayFineControl)
        self.subVisuPanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.subVisuPanel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.subVisuPanel.setObjectName("subVisuPanel")


        self.layoutControlGrid3 = QtWidgets.QGridLayout(self.subVisuPanel)
        self.layoutControlGrid3.setObjectName("layoutControlGrid3")


        self.lineEdSubSampVisu = QtWidgets.QLineEdit(parent=self.subVisuPanel)
        self.lineEdSubSampVisu.setMaximumSize(QtCore.QSize(80, 16777215))
        self.lineEdSubSampVisu.setObjectName("lineEdSubSampVisu")


        self.layoutControlGrid3.addWidget(self.lineEdSubSampVisu, 0, 0, 1, 1)
        self.lineEdIDSelY = QtWidgets.QLineEdit(parent=self.subVisuPanel)
        self.lineEdIDSelY.setObjectName("lineEdIDSelY")


        self.layoutControlGrid3.addWidget(self.lineEdIDSelY, 1, 0, 1, 1)
        self.lineEdIDSelPar = QtWidgets.QLineEdit(parent=self.subVisuPanel)
        self.lineEdIDSelPar.setObjectName("lineEdIDSelPar")


        self.layoutControlGrid3.addWidget(self.lineEdIDSelPar, 1, 1, 1, 1)
        self.pushAutoSubSamp = QtWidgets.QPushButton(parent=self.subVisuPanel)
        self.pushAutoSubSamp.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pushAutoSubSamp.setObjectName("pushAutoSubSamp")


        self.layoutControlGrid3.addWidget(self.pushAutoSubSamp, 0, 1, 1, 1)
        self.layoutDisplayFineControl.addWidget(self.subVisuPanel, 1, 1, 1, 1)
        self.dimensionSelectionPanel = QtWidgets.QFrame(parent=self.displayFineControl)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dimensionSelectionPanel.sizePolicy().hasHeightForWidth())
        self.dimensionSelectionPanel.setSizePolicy(sizePolicy)
        self.dimensionSelectionPanel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.dimensionSelectionPanel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.dimensionSelectionPanel.setObjectName("dimensionSelectionPanel")


        self.layoutControlGrid1 = QtWidgets.QGridLayout(self.dimensionSelectionPanel)
        self.layoutControlGrid1.setObjectName("layoutControlGrid1")


        self.selectDimR = QtWidgets.QComboBox(parent=self.dimensionSelectionPanel)
        self.selectDimR.setObjectName("selectDimR")
        # self.selectDimR.addItem("")
        # self.selectDimR.addItem("")

        self.layoutControlGrid1.addWidget(self.selectDimR, 0, 0, 1, 1)
        self.selectDim2 = QtWidgets.QComboBox(parent=self.dimensionSelectionPanel)
        self.selectDim2.setObjectName("selectDim2")


        self.layoutControlGrid1.addWidget(self.selectDim2, 1, 1, 1, 1)
        self.selectDimP = QtWidgets.QComboBox(parent=self.dimensionSelectionPanel)
        self.selectDimP.setObjectName("selectDimP")


        self.layoutControlGrid1.addWidget(self.selectDimP, 0, 1, 1, 1)
        self.selectDim1 = QtWidgets.QComboBox(parent=self.dimensionSelectionPanel)
        self.selectDim1.setObjectName("selectDim1")


        self.layoutControlGrid1.addWidget(self.selectDim1, 1, 0, 1, 1)
        self.layoutDisplayFineControl.addWidget(self.dimensionSelectionPanel, 0, 0, 1, 1)
        self.layoutDisplayControl.addWidget(self.displayFineControl)
        self.layoutDisplaySubPanel.addWidget(self.displayControl)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.layoutDisplaySubPanel.addItem(spacerItem1)
        self.inferenceView = QtWidgets.QFrame(parent=self.subDisplayPanel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inferenceView.sizePolicy().hasHeightForWidth())
        self.inferenceView.setSizePolicy(sizePolicy)
        self.inferenceView.setMinimumSize(QtCore.QSize(280, 0))
        self.inferenceView.setMaximumSize(QtCore.QSize(500, 16777215))
        self.inferenceView.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.inferenceView.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.inferenceView.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.inferenceView.setObjectName("inferenceView")


        self.labelResults = QtWidgets.QLabel(parent=self.inferenceView)
        self.labelResults.setGeometry(QtCore.QRect(10, 10, 120, 20))
        self.labelResults.setStyleSheet("text-decoration: underline;\n"
"font: 9pt \"Sans Serif\";")
        self.labelResults.setObjectName("labelResults")


        self.labelResult1 = QtWidgets.QLabel(parent=self.inferenceView)
        self.labelResult1.setGeometry(QtCore.QRect(10, 40, 200, 17))
        self.labelResult1.setStyleSheet("")
        self.labelResult1.setObjectName("labelResult1")


        self.labelResult2 = QtWidgets.QLabel(parent=self.inferenceView)
        self.labelResult2.setGeometry(QtCore.QRect(10, 65, 200, 17))
        self.labelResult2.setStyleSheet("")
        self.labelResult2.setObjectName("labelResult2")


        self.labelResult3 = QtWidgets.QLabel(parent=self.inferenceView)
        self.labelResult3.setGeometry(QtCore.QRect(10, 90, 200, 17))
        self.labelResult3.setStyleSheet("")
        self.labelResult3.setObjectName("labelResult3")


        self.layoutDisplaySubPanel.addWidget(self.inferenceView)
        self.layoutPanel.addWidget(self.subDisplayPanel)
        self.layoutCentral.addWidget(self.displayPanel)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 22))
        self.menubar.setObjectName("menubar")


        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")

        
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabViewPanel.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bayesian Inference"))
        self.labelTitle.setText(_translate("MainWindow", "Bayesian Inference GUI"))
        self.labelDataSelect.setText(_translate("MainWindow", "Data Set Selection"))
        self.selectDataCase.setPlaceholderText(_translate("MainWindow", "Data-Set"))
        self.selectDataCase.setItemText(0, _translate("MainWindow", "Polynomial"))
        self.selectDataCase.setItemText(1, _translate("MainWindow", "Housing"))
        self.pushCustomCase.setText(_translate("MainWindow", "Custom"))
        self.pushImportData.setText(_translate("MainWindow", "Import Data"))
        self.lineEdCV.setPlaceholderText(_translate("MainWindow", "Cross Validation"))
        self.lineEdDataSize.setPlaceholderText(_translate("MainWindow", "Data set size"))
        self.labelRegModel.setText(_translate("MainWindow", "Regressor Model Selection (deterministic)"))
        self.selectRegMod.setItemText(0, _translate("MainWindow", "Linear Polynomial"))
        self.selectRegMod.setItemText(1, _translate("MainWindow", "ElasticNet"))
        self.selectRegMod.setItemText(2, _translate("MainWindow", "SVR"))
        self.selectRegMod.setItemText(3, _translate("MainWindow", "RandomForest"))
        self.pushParamReg.setText(_translate("MainWindow", "Parameters"))
        self.pushFitReg.setText(_translate("MainWindow", "Fit Model"))
        self.labelBayesModel.setText(_translate("MainWindow", "Bayesian Model Selection"))
        self.selectBayesModel.setItemText(0, _translate("MainWindow", "Model Form"))
        self.selectBayesModel.setItemText(1, _translate("MainWindow", "Linear Gaussian Fixed StdDev"))
        self.selectBayesModel.setItemText(2, _translate("MainWindow", "Linear Gaussian"))
        self.selectParamTune.setItemText(0, _translate("MainWindow", "Parameter"))
        self.selectDistType.setItemText(0, _translate("MainWindow", "Type"))
        self.selectDistType.setItemText(1, _translate("MainWindow", "Normal"))
        self.selectDistType.setItemText(2, _translate("MainWindow", "Uniform"))
        self.selectDistType.setItemText(3, _translate("MainWindow", "Half-Normal"))
        self.pushParamAuto.setText(_translate("MainWindow", "Auto"))
        self.lineEdpar1.setPlaceholderText(_translate("MainWindow", "Hyp1"))
        self.lineEdpar2.setPlaceholderText(_translate("MainWindow", "Hyp2"))
        self.labelInferenceConfig.setText(_translate("MainWindow", "Inference Configuration"))
        self.lineEdMC1.setPlaceholderText(_translate("MainWindow", "N_MCMC"))
        self.lineEdMC3.setPlaceholderText(_translate("MainWindow", "N_tune"))
        self.radioStepType1.setText(_translate("MainWindow", "Block Step"))
        self.radioStepType2.setText(_translate("MainWindow", "Single Step"))
        self.lineEdMC2.setPlaceholderText(_translate("MainWindow", "N_thin"))
        self.pushCompute.setText(_translate("MainWindow", "Compute / Refresh"))
        self.pushExport.setText(_translate("MainWindow", "Export Results"))
        self.tabViewPanel.setTabText(self.tabViewPanel.indexOf(self.tab), _translate("MainWindow", "Posterior prediction"))
        self.tabViewPanel.setTabText(self.tabViewPanel.indexOf(self.tab_2), _translate("MainWindow", "Posterior parameters"))
        self.tabViewPanel.setTabText(self.tabViewPanel.indexOf(self.tab_3), _translate("MainWindow", "Fit / Error"))
        self.tabViewPanel.setTabText(self.tabViewPanel.indexOf(self.tab_4), _translate("MainWindow", "Chains"))
        self.labelDisplay.setText(_translate("MainWindow", "Display Control"))
        self.checkLegend1.setText(_translate("MainWindow", "Train Data"))
        self.checkLegend2.setText(_translate("MainWindow", "Test Data"))
        self.checkLegend3.setText(_translate("MainWindow", "Post MAP"))
        self.checkLegend4.setText(_translate("MainWindow", "Post Dist Y"))
        self.checkLegend5.setText(_translate("MainWindow", "Regressor"))
        self.lineEdXmax.setPlaceholderText(_translate("MainWindow", "xmax"))
        self.pushAutoRange.setText(_translate("MainWindow", "Auto"))
        self.labelRange.setText(_translate("MainWindow", "Range"))
        self.lineEdXmin.setPlaceholderText(_translate("MainWindow", "xmin"))
        self.lineEdYmin.setPlaceholderText(_translate("MainWindow", "ymin"))
        self.lineEdYmax.setPlaceholderText(_translate("MainWindow", "ymax"))
        self.lineEdSubSampVisu.setPlaceholderText(_translate("MainWindow", "SubSamp"))
        self.lineEdIDSelY.setPlaceholderText(_translate("MainWindow", "ID Y"))
        self.lineEdIDSelPar.setPlaceholderText(_translate("MainWindow", "ID param"))
        self.pushAutoSubSamp.setText(_translate("MainWindow", "Rng"))
        self.selectDimR.setPlaceholderText(_translate("MainWindow", "Dim X"))
        # self.selectDimR.setItemText(0, _translate("MainWindow", "0"))
        # self.selectDimR.setItemText(1, _translate("MainWindow", "1"))
        self.selectDim2.setPlaceholderText(_translate("MainWindow", "Scatter2"))
        self.selectDimP.setPlaceholderText(_translate("MainWindow", "Grid"))
        self.selectDim1.setPlaceholderText(_translate("MainWindow", "Scatter1"))
        self.labelResults.setText(_translate("MainWindow", "Inference Results"))
        self.labelResult1.setText(_translate("MainWindow", "Inference details"))
        self.labelResult2.setText(_translate("MainWindow", "MAP / Model SnapShot"))
        self.labelResult3.setText(_translate("MainWindow", "Fit Quality"))
