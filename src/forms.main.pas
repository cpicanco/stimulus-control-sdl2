{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit forms.main;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Dialogs, StdCtrls, ExtCtrls,
  IniPropStorage, ComCtrls, Menus, PropertyStorage;

type

  { TFormMain }

  TFormMain = class(TForm)
    ButtonRunNextSession: TButton;
    ButtonOpenSpeechValidation: TButton;
    ButtonTestCondition: TButton;
    ButtonMisc: TButton;
    ButtonNewParticipant: TButton;
    ButtonRunInterruptedSession: TButton;
    ButtonRunNewSession: TButton;
    ComboBoxDesignFolder: TComboBox;
    ComboBoxParticipant: TComboBox;
    IniPropStorage1: TIniPropStorage;
    LabelSessionEndCriteria: TLabel;
    LabelLastSessionName: TLabel;
    ListBoxCondition: TListBox;
    MenuItemShowWordsPerCycle: TMenuItem;
    MenuItemOutputWordsPerCyle: TMenuItem;
    MenuItemCyclesFromTemplate: TMenuItem;
    MenuItemConvertDesignFile: TMenuItem;
    MenuItemRemoveParticipant: TMenuItem;
    MenuItemCopyPNGFiles: TMenuItem;
    OpenDialog1: TOpenDialog;
    OpenDialog2: TOpenDialog;
    Panel1: TPanel;
    PopupMenuParticipants: TPopupMenu;
    PopupMenuMisc: TPopupMenu;
    ProgressBar: TProgressBar;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    procedure ButtonOpenSpeechValidationClick(Sender: TObject);
    procedure ButtonRunNextSessionClick(Sender: TObject);
    procedure ButtonTestConditionClick(Sender: TObject);
    procedure ButtonRunInterruptedSessionClick(Sender: TObject);
    procedure ButtonMiscClick(Sender: TObject);
    procedure ButtonRunNewSessionClick(Sender: TObject);
    procedure ButtonNewParticipantClick(Sender: TObject);
    procedure BeginSession(Sender: TObject);
    procedure ComboBoxDesignFolderEditingDone(Sender: TObject);
    procedure ComboBoxParticipantEditingDone(Sender: TObject);
    procedure EndSession(Sender : TObject);
    procedure CloseSDLApp(Sender : TObject);
    procedure FormCreate(Sender: TObject);
    procedure IniPropStorage1RestoreProperties(Sender: TObject);
    procedure IniPropStorage1StoredValues0Restore(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorage1StoredValues0Save(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorage1StoredValues1Restore(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorage1StoredValues1Save(Sender: TStoredValue;
      var Value: TStoredType);
    procedure MenuItemConvertDesignFileClick(Sender: TObject);
    procedure MenuItemCopyPNGFilesClick(Sender: TObject);
    procedure MenuItemCyclesFromTemplateClick(Sender: TObject);
    procedure MenuItemOutputWordsPerCyleClick(Sender: TObject);
    procedure MenuItemRemoveParticipantClick(Sender: TObject);
    procedure MenuItemShowWordsPerCycleClick(Sender: TObject);
  private
    FSessionName : string;
    procedure HitCriteriaAtSessionEnd(Sender : TObject);
    procedure NotHitCriteriaAtSessionEnd(Sender : TObject);
    function AssignGlobalVariables : Boolean;
    procedure ToogleControlPanelEnabled(AException: TComponent = nil);
    function ParticipantFolderName : string;
    function SessionName : string;
    function SetupFolders : Boolean;
    function Validated : Boolean;
    procedure RunSession;
  public
    procedure CreateNewConfigurationFile;
  end;

var
  FormMain: TFormMain;

implementation

{$R *.lfm}

uses
  FileUtil
  , Graphics
  , common.helpers
  , forms.main.misc
  , forms.test.session.endcriteria
  , forms.speechvalidation
  , session
  , session.information
  , session.parameters.global
  , session.pool
  , session.fileutils
  , session.csv.experiments
  , session.design.conversion
  , sdl.app
  , sdl.app.controller.manager
  , sdl.app.grids.types
  , sdl.app.testmode
  , eye.tracker
  , ui.backup
  , picanco.experiments.images
  , picanco.experiments.output
  , sdl.colors
  ;

{ ToDo: show next designed session of selected participant.
        for data in data folder get next session of last session}

{ ToDo: point-and-click design tools }

{ TFormMain }

procedure TFormMain.ButtonRunNewSessionClick(Sender: TObject);
begin
  FormMisc.Initialize;
  if not AssignGlobalVariables then Exit;
  if not Validated then Exit;

  ToogleControlPanelEnabled;
  CreateNewConfigurationFile;
  RunSession;
end;

procedure TFormMain.ButtonNewParticipantClick(Sender: TObject);
var
  LNewParticipant : string;
begin
  with ComboBoxParticipant do begin
    LNewParticipant :=
      InputBox(Pool.AppName,
      'Nome: mínimo de 3 caracteres',
      '');

    if LNewParticipant.IsEmpty or (Length(LNewParticipant) < 3) then Exit;
    Items.Append(LNewParticipant);
    ItemIndex := Items.Count-1;
    MenuItemCopyPNGFilesClick(ButtonNewParticipant);
  end;
end;

procedure TFormMain.ButtonRunInterruptedSessionClick(Sender: TObject);
begin
  SetupFolders; // todo: pass filename id of loaded file into session.counters.loadfromfile
  OpenDialog1.InitialDir := Pool.BaseDataPath;
  if OpenDialog1.Execute then begin
    Pool.ConfigurationFilename := LoadConfigurationFile(OpenDialog1.FileName);
    FSessionName := LoadInformationFromFile(OpenDialog1.FileName).SessionName;
    ShowMessage(FSessionName);
    ProgressBar.Max := 1;
    ProgressBar.StepIt;
    ProgressBar.Visible := True;

    RunSession;
  end;
end;

procedure TFormMain.ButtonTestConditionClick(Sender: TObject);
begin
  FormEndCriteriaTest.Show;
end;

procedure TFormMain.ButtonOpenSpeechValidationClick(Sender: TObject);
begin
  FormSpeechValidationQueue.Show;
end;

procedure TFormMain.ButtonRunNextSessionClick(Sender: TObject);
var
  LInformation : TInformation;
  LCondition : integer;

  procedure ShowErrorMessage;
  begin
    ShowMessage('Não foi possível executar a sessão seguinte');
  end;

begin
  with ListBoxCondition do begin
    if ComboBoxParticipant.Items.Count > 0 then begin
      SetupFolders;
      if LastValidBaseFilenameFileExists then begin
        LInformation := LoadInformationFromFile(GetLastValidInformationFile);
      end else begin
        //ShowMessage('Guessing last valid information file');
        LInformation := LoadInformationFromFile(GuessLastValidInformationFile);
      end;

      LCondition := ListBoxCondition.Items.IndexOf(LInformation.SessionName);
      if LCondition > -1 then begin
        ItemIndex := LCondition;
        if ItemIndex < Items.Count -1 then begin
          ItemIndex := ItemIndex + 1;
          ButtonRunNewSessionClick(Sender);
        end else begin
          ShowErrorMessage;
        end;
      end else begin
        ShowErrorMessage;
      end;
    end else begin
      ShowErrorMessage;
    end;
  end;
end;

procedure TFormMain.ButtonMiscClick(Sender: TObject);
begin
  FormMisc.ShowModal;
end;

procedure TFormMain.BeginSession(Sender: TObject);
begin
  if Assigned(EyeTracker) then begin
    EyeTracker.SetDataFilename(Pool.BaseDataPath + Pool.BaseFilename);
    EyeTracker.StartRecording;
  end;
end;

procedure TFormMain.ComboBoxDesignFolderEditingDone(Sender: TObject);
begin
  with ComboBoxDesignFolder do begin
    if ItemIndex > 0 then begin
      if ComboBoxDesignFolder.Items.Count > 0 then begin
        ListBoxCondition.Clear;
        Pool.DesignBasePath := Items[ItemIndex];
        SaveProtocolIndex(ParticipantFolderName, ItemIndex);
        GetDesignFilesFor(ListBoxCondition.Items);
      end;
    end;
  end;
end;

procedure TFormMain.ComboBoxParticipantEditingDone(Sender: TObject);
var
  LInformation : TInformation;
  LCondition : integer;
begin
  if ComboBoxParticipant.Items.Count > 0 then begin
    SetupFolders;
    ComboBoxDesignFolder.ItemIndex := LoadProtocolIndex(ParticipantFolderName);
    ComboBoxDesignFolderEditingDone(ComboBoxParticipant);
    //IniPropStorageProtocol.Save;
    //LConfiguration := ConcatPaths([
    //  Pool.ConfigurationsRootBasePath,
    //  ParticipantFolderName, 'protocol.ini']);
    //IniPropStorageProtocol.IniFileName := LConfiguration;
    //if FileExists(LConfiguration) then begin
    //  IniPropStorageProtocol.Restore;
    //end;

    //FormMisc.IniPropStorage1.Save;
    //LConfiguration := ConcatPaths([
    //  Pool.ConfigurationsRootBasePath,
    //  ParticipantFolderName, 'configurations_global.ini']);
    //FormMisc.IniPropStorage1.IniFileName := LConfiguration;
    //if FileExists(LConfiguration) then begin
    //  FormMisc.IniPropStorage1.Restore;
    //end;

    if LastValidBaseFilenameFileExists then begin
      LInformation := LoadInformationFromFile(GetLastValidInformationFile);
    end else begin
      //ShowMessage('Guessing last valid information file');
      LInformation := LoadInformationFromFile(GuessLastValidInformationFile);
    end;

    if LInformation.Basename.IsEmpty then begin
      LabelLastSessionName.Caption := '?';
      LabelSessionEndCriteria.Caption := '?';
    end else begin
      LabelLastSessionName.Caption := LInformation.SessionName;
      LabelSessionEndCriteria.Caption := LInformation.SessionResult;
      LCondition := ListBoxCondition.Items.IndexOf(LInformation.SessionName);
      if LCondition > -1 then begin
        ListBoxCondition.ItemIndex := LCondition;
      end;
    end;
  end;
end;

procedure TFormMain.EndSession(Sender: TObject);
begin

end;

procedure TFormMain.CloseSDLApp(Sender: TObject);
var
  LPoint : TPoint;
begin
  LPoint := Point(ButtonRunNextSession.Left, ButtonRunNextSession.Top);
  Mouse.CursorPos.SetLocation(ClientToScreen(LPoint));

  FormMisc.Finalize;
  SDLSession.Free;
  SDLApp.Free;

  if Assigned(EyeTracker) then begin
    EyeTracker.StopRecording;
    FinalizeEyeTracker;
  end;

  FreeConfigurationFile;
  Controllers.Free;
  ToogleControlPanelEnabled;
  ProgressBar.Visible := False;
  FSessionName := '';
  ButtonRunNextSession.SetFocus;
end;

procedure TFormMain.FormCreate(Sender: TObject);
begin
  FSessionName := '';
end;

procedure TFormMain.IniPropStorage1RestoreProperties(Sender: TObject);
begin
  SetupFolders;
end;

procedure TFormMain.IniPropStorage1StoredValues0Restore(
  Sender: TStoredValue; var Value: TStoredType);
var
  LValue : integer;
begin
  GetDesignFoldersFor(ComboBoxDesignFolder.Items);
  with Pool, ComboBoxDesignFolder do begin
    LValue := Value.ToInteger;
    if (LValue < Items.Count) and (LValue <> -1) then begin
      DesignBasePath := Items[LValue];
    end;
  end;
end;

procedure TFormMain.IniPropStorage1StoredValues0Save(
  Sender: TStoredValue; var Value: TStoredType);
begin
  SaveProtocolIndex(ParticipantFolderName, ComboBoxDesignFolder.ItemIndex);
  Value := ComboBoxDesignFolder.ItemIndex.ToString;
end;

procedure TFormMain.IniPropStorage1StoredValues1Restore(
  Sender: TStoredValue; var Value: TStoredType);
var
  LValue : integer;
begin
  GetDesignFilesFor(ListBoxCondition.Items);
  LValue := Value.ToInteger;
  if LValue < ListBoxCondition.Count then begin
    ListBoxCondition.ItemIndex := LValue;
  end;
end;

procedure TFormMain.IniPropStorage1StoredValues1Save(
  Sender: TStoredValue; var Value: TStoredType);
begin
  Value := ListBoxCondition.ItemIndex.ToString;
end;

procedure TFormMain.MenuItemConvertDesignFileClick(Sender: TObject);
begin
  CovertToSingleFilename;
end;

procedure TFormMain.MenuItemCopyPNGFilesClick(Sender: TObject);
begin
  SetupFolders;
  E1CopyRandomImagesToParticipantFolder;
end;

procedure TFormMain.MenuItemCyclesFromTemplateClick(Sender: TObject);
begin
  CyclesFromTemplate;
end;

procedure TFormMain.MenuItemOutputWordsPerCyleClick(Sender: TObject);
begin
  PrintWordsPerCycle;
end;

procedure TFormMain.MenuItemRemoveParticipantClick(Sender: TObject);
begin
  with ComboBoxParticipant do
    Items.Delete(ItemIndex);
end;

procedure TFormMain.MenuItemShowWordsPerCycleClick(Sender: TObject);
begin
  ShowWordsPerCycle;
end;

procedure TFormMain.HitCriteriaAtSessionEnd(Sender: TObject);
const
  LResult = 'Critério atingido';
begin
  SetSessionResult(LResult);
  with ListBoxCondition do begin
    LabelLastSessionName.Caption := Items[ItemIndex];
  end;
  LabelSessionEndCriteria.Color := clGreen;
  LabelSessionEndCriteria.Caption := LResult;
end;

procedure TFormMain.NotHitCriteriaAtSessionEnd(Sender: TObject);
const
  LResult = 'Critério não atingido';
begin
  SetSessionResult(LResult);
  with ListBoxCondition do begin
    LabelLastSessionName.Caption := Items[ItemIndex];
  end;
  LabelSessionEndCriteria.Color := Graphics.clRed;
  LabelSessionEndCriteria.Caption := LResult;
end;

function TFormMain.AssignGlobalVariables: Boolean;
begin
  Result := False;
  TestMode := FormMisc.CheckBoxTestMode.Checked;

  GlobalTrialParameters.Cursor := 1;

  with GlobalTrialParameters, FormMisc.FloatSpinEditScreenWidth do
    ScreenInCentimeters := Value;

  with GlobalTrialParameters, FormMisc.FloatSpinEditCellsSize do
    CellsSizeInCentimenter := Value;

  with GlobalTrialParameters, FormMisc.CheckBoxHideMouse do
    HideMouse := Checked;

  with GlobalTrialParameters, FormMisc.SpinEditAprilTagsSize do
    MarkerSize := Value;

  with GlobalTrialParameters,
       FormMisc.CheckBoxShowModalFormForSpeechResponses do
    ShowModalFormForSpeechResponses := Checked;

  with GlobalTrialParameters, FormMisc.ComboBoxShouldRestartAt do
    ShouldRestartAtBlockStart := ItemIndex = 0;

  with GlobalTrialParameters, FormMisc.ComboBoxAudioPromptForText do begin
    if ItemIndex > -1 then begin
      AudioPromptForText := Items[ItemIndex];
    end else begin
      ShowMessage('Escolha um prompt de texto nas configurações.');
      Exit;
    end;
  end;

  with GlobalTrialParameters, FormMisc.ComboBoxFontName do begin
    if ItemIndex > -1 then begin
      FontName := Items[ItemIndex];
    end else begin
      ShowMessage('Escolha uma fonte para o texto nas configurações.');
      Exit;
    end;
  end;

  with FormMisc.PanelFontColor do begin
    sdl.colors.clDefaultBackground := ColorToSDLColor(Color);
    sdl.colors.clBackground := ColorToSDLColor(Color);
    sdl.colors.clFontColor := ColorToSDLColor(Font.Color);
  end;

  with GlobalTrialParameters, FormMisc.CheckBoxUseRemoteServer do begin
    UseRemoteServer := Checked;
  end;

  with GlobalTrialParameters, FormMisc.CheckBoxUseGazeAsInput do begin
    UseGazeAsInput := Checked;
  end;

  with GlobalTrialParameters, FormMisc.CheckBoxSimultaneousMTS do begin
    SimultaneousMTS := Checked;
  end;

  with GlobalTrialParameters, FormMisc.SpinEditFontSize do
    FontSize := Value;

  with GlobalTrialParameters, FormMisc.SpinEditRecordingSeconds do
    RecordingSeconds := Value;

  with GlobalTrialParameters, FormMisc.SpinEditInterTrialInterval do
    InterTrialInterval := Value;

  with GlobalTrialParameters, FormMisc.SpinEditLimitedHold do
    LimitedHold := Value.MinutesToMiliseconds;

  with GlobalTrialParameters, FormMisc.SpinEditAudioLoopInterval do
    AudioLoopInterval := Value;

  with GlobalTrialParameters, FormMisc.SpinEditDefaultAudioLoops do
    DefaultAudioLoops := Value;

  with GlobalTrialParameters, FormMisc.SpinEditTimeOut do
    TimeOutInterval := Value;

  with GlobalTrialParameters, FormMisc.ComboBoxAudioFolder do begin
    if ItemIndex > -1 then begin
      Pool.AudioBasePath := Items[ItemIndex];
    end else begin
      ShowMessage('Escolha uma pasta de audio nas configurações.');
      Exit;
    end;
  end;

  with GlobalTrialParameters, FormMisc.ComboBoxFixedSamplePosition do begin
    GridOrientation := goCustom;
    case ItemIndex of
      1: begin // centralize sample, use 4 corners for comparisons
        FixedSamplePosition := 4;
        FixedComparisonPosition := 4;
        SetLength(ComparisonPositions, 4);
        ComparisonPositions[0] := 0;
        ComparisonPositions[1] := 2;
        ComparisonPositions[2] := 6;
        ComparisonPositions[3] := 8;
      end
      else begin // upper sample, use 3 bottom positions for comparisons
        FixedSamplePosition := 1;
        SetLength(ComparisonPositions, 3);
        ComparisonPositions[0] := 6;
        ComparisonPositions[1] := 7;
        ComparisonPositions[2] := 8;
      end;
    end;
  end;
  Result := True;
end;

procedure TFormMain.ToogleControlPanelEnabled(AException: TComponent);
var
  i: Integer;
begin
  for i := 0 to ComponentCount -1 do begin
    if (Components[i] is TControl) and
       (Components[i] <> AException) then begin
      TControl(Components[i]).Enabled := not TControl(Components[i]).Enabled;
    end;
  end;
end;

function TFormMain.ParticipantFolderName: string;
begin
  if ComboBoxParticipant.Items.Count > 0 then begin
    if ComboBoxParticipant.ItemIndex > -1 then begin
      Pool.Counters.Subject := ComboBoxParticipant.ItemIndex;
      Result := Pool.Counters.Subject.ToString +'-'+
          ComboBoxParticipant.Items[Pool.Counters.Subject] +
          DirectorySeparator;
    end;
  end;
end;

function TFormMain.SessionName: string;
begin
  if FSessionName.IsEmpty then begin
    with ListBoxCondition do begin
      Result := Items[ItemIndex];
    end;
  end else begin
    Result := FSessionName;
  end;
end;

function TFormMain.SetupFolders: Boolean;
begin
  Pool.ImageBasePath := ParticipantFolderName;
  Pool.BaseDataPath :=
    ConcatPaths([Pool.DataRootBasePath, ParticipantFolderName]);

  Pool.DataResponsesBasePath :=
    ConcatPaths([Pool.BaseDataPath, Pool.ResponsesBasePath]);

  Result :=
    ForceDirectories(Pool.BaseDataPath) and
    ForceDirectories(Pool.DataResponsesBasePath);
end;

function TFormMain.Validated: Boolean;
  function SetupParticipantID : Boolean;
  var
    LParticipantID: TStringList;
    LIDFile : string;
    LID : string;
  begin
    LIDFile := Pool.DataRootBasePath + ParticipantFolderName + 'ID';
    LID := Pool.Counters.Subject.ToString;
    LParticipantID := TStringList.Create;
    try
      if FileExists(LIDFile) then begin
        LParticipantID.LoadFromFile(LIDFile);
        if LID = LParticipantID[0] then begin
          Result := True;
        end else begin
          Result := False;
          ShowMessage(
            'Inconsistência:' + LineEnding +
            'LID:' + LID + ' <> ' + LParticipantID[0]);

        end;
      end else begin
        LParticipantID.Clear;
        LParticipantID.Append(LID);
        try
          Result := True;
          LParticipantID.SaveToFile(LIDFile);
        except
          on EFilerError do begin
            Result := False;
          end;
        end;

      end;
    finally
      LParticipantID.Free;
    end;
  end;

begin
  Result := False;

  with FormMisc.ComboBoxMonitor do begin
    if FormMisc.CheckBoxForceLastMonitor.Checked then begin
      if Items.Count > 0 then begin
        ItemIndex := Items.Count-1;
      end;
    end else begin
      if ItemIndex = -1 then begin
        if Items.Count > 0 then begin
          ItemIndex := Items.Count-1;
          ShowMessage('O último monitor foi selecionado: ' + Items[ItemIndex]);
        end else begin
          ShowMessage('Nenhum monitor foi reconhecido.');
          Exit;
        end;
      end;
    end;
  end;

  if Pool.ConfigurationFilename.IsEmpty then begin
    with ListBoxCondition do begin
      if ItemIndex > -1 then begin
        ShowMessage(
          'Uma nova sessão será criada:' + LineEnding + Items[ItemIndex]);
      end else begin
        Exit;
      end;
    end;
  end;

  if ComboBoxParticipant.Items.Count = 0 then begin
    ShowMessage('Crie um novo participante.');
    Exit;
  end;

  if ComboBoxParticipant.ItemIndex < 0 then begin
    ShowMessage('Escolha um participante.');
    Exit;
  end;

  if not SetupFolders then begin
    ShowMessage('Não foi possível criar a estrutura de diretórios.');
    Exit;
  end;

  if not SetupParticipantID then begin
    ShowMessage('Não foi possível criar o arquivo ID do participante.');
    Exit;
  end;

  Result := True;
end;

procedure TFormMain.RunSession;
begin
  SDLApp := TSDLApplication.Create(@Pool.AppName[1]);
  SDLApp.SetupVideo(FormMisc.ComboBoxMonitor.ItemIndex);
  //SDLApp.PrintRendererSetup;
  SDLApp.SetupAudio;
  SDLApp.SetupText;
  SDLApp.OnClose := @CloseSDLApp;
  SDLApp.ShowMarkers := FormMisc.CheckBoxShowMarkers.Checked;

  Controllers := TControllerManager.Create;
  Controllers.CreateController(FormMisc.ComboBoxController.ItemIndex);

  Pool.App := SDLApp;

  InitializeEyeTracker(FormMisc.ComboBoxEyeTracker.ItemIndex);

  Pool.SessionName := SessionName;
  Pool.ParticipantName := ParticipantFolderName;

  SDLSession := TSession.Create(Self);
  SDLSession.OnBeforeStart := @BeginSession;
  SDLSession.OnEndSession  := @EndSession;
  SDLSession.OnHitCriteriaAtSessionEnd := @HitCriteriaAtSessionEnd;
  SDLSession.OnNotHitCriteriaAtSessionEnd := @NotHitCriteriaAtSessionEnd;
  SDLSession.Play;

  SDLApp.Run;
end;

procedure TFormMain.CreateNewConfigurationFile;
var
  LFilename: String;
begin
  if ListBoxCondition.Items.Count = 0 then begin
    ShowMessage('A pasta de parâmetros (design) está vazia.');
    Exit;
  end;

  if ListBoxCondition.ItemIndex = -1 then begin
    ShowMessage('Escolha um arquivo com os parâmetro da sessão.');
    Exit;
  end else begin
    with ListBoxCondition do begin
      LFilename := Items[ItemIndex];
    end;
  end;

  ToogleControlPanelEnabled(ProgressBar);
  Pool.ConfigurationFilename := MakeConfigurationFile(LFilename);
  ProgressBar.Visible := True;
  ToogleControlPanelEnabled(ProgressBar);
end;

end.

