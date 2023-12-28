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

  { TFormBackground }

  TFormBackground = class(TForm)
    ButtonMisc: TButton;
    ButtonNewParticipant: TButton;
    ButtonLoadConfigurationFile: TButton;
    ButtonNewConfigurationFile: TButton;
    ButtonRunSession: TButton;
    ComboBoxCondition: TComboBox;
    ComboBoxDesignFolder: TComboBox;
    ComboBoxParticipant: TComboBox;
    IniPropStorage1: TIniPropStorage;
    LabelContact: TLabel;
    MenuItemRemoveParticipant: TMenuItem;
    MenuItemCopyPNGFiles: TMenuItem;
    OpenDialog1: TOpenDialog;
    Panel1: TPanel;
    PopupMenuParticipants: TPopupMenu;
    PopupMenuMisc: TPopupMenu;
    ProgressBar: TProgressBar;
    procedure ButtonLoadConfigurationFileClick(Sender: TObject);
    procedure ButtonMiscClick(Sender: TObject);
    procedure ButtonNewConfigurationFileClick(Sender: TObject);
    procedure ButtonNewParticipantClick(Sender: TObject);
    procedure ButtonRunSessionClick(Sender: TObject);
    procedure BeginSession(Sender: TObject);
    procedure ComboBoxDesignFolderEditingDone(Sender: TObject);
    procedure EndSession(Sender : TObject);
    procedure CloseSDLApp(Sender : TObject);
    procedure IniPropStorage1RestoreProperties(Sender: TObject);
    procedure IniPropStorage1StoredValues0Restore(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorage1StoredValues0Save(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorage1StoredValues1Restore(Sender: TStoredValue;
      var Value: TStoredType);
    procedure IniPropStorage1StoredValues1Save(Sender: TStoredValue;
      var Value: TStoredType);
    procedure MenuItemCopyPNGFilesClick(Sender: TObject);
    procedure MenuItemRemoveParticipantClick(Sender: TObject);
  private
    //FEyeLink : TEyeLink;
    procedure AssignGlobalVariables;
    procedure ToogleControlPanelEnabled(AException: TComponent = nil);
    function ParticipantFolderName : string;
    function SessionName : string;
    function SetupFolders : Boolean;
    function Validated : Boolean;
  public

  end;

var
  FormBackground: TFormBackground;

implementation

{$R *.lfm}

uses
  FileUtil
  , common.helpers
  , forms.main.misc
  , session
  , session.parameters.global
  , session.pool
  , session.loggers
  , session.fileutils
  , session.csv.experiments
  , sdl.app
  , sdl.app.controller.manager
  , sdl.app.grids.types
  , sdl.app.testmode
  , eye.tracker
  , picanco.experiments.images
  ;

{ ToDo: show next designed session of selected participant.
        for data in data folder get next session of last session}

{ TFormBackground }

procedure TFormBackground.ButtonRunSessionClick(Sender: TObject);
begin
  if not Validated then Exit;
  AssignGlobalVariables;
  ToogleControlPanelEnabled;

  SDLApp := TSDLApplication.Create(@Pool.AppName[1]);
  SDLApp.SetupVideo(FormMisc.ComboBoxMonitor.ItemIndex);
  SDLApp.SetupAudio;
  SDLApp.SetupText;
  SDLApp.OnClose := @CloseSDLApp;
  SDLApp.ShowMarkers := FormMisc.CheckBoxShowMarkers.Checked;

  Controllers := TControllerManager.Create;
  Controllers.CreateController(FormMisc.ComboBoxController.ItemIndex);

  Pool.App := SDLApp;

  InitializeEyeTracker(FormMisc.ComboBoxEyeTracker.ItemIndex);

  SDLSession := TSession.Create(Self);
  SDLSession.OnBeforeStart := @BeginSession;
  SDLSession.OnEndSession  := @EndSession;
  SDLSession.Play;

  SDLApp.Run;
end;

procedure TFormBackground.ButtonNewConfigurationFileClick(Sender: TObject);
var
  LFilename : string;
begin
  AssignGlobalVariables;
  if ComboBoxCondition.Items.Count = 0 then begin
    ShowMessage('A pasta de parâmetros (design) está vazia.');
    Exit;
  end;
  if ComboBoxCondition.ItemIndex = -1 then begin
    ShowMessage('Escolha um parâmetro.');
    Exit;
  end else begin
    with ComboBoxCondition do begin
      LFilename := Items[ItemIndex];
    end;
  end;
  ToogleControlPanelEnabled(ProgressBar);
  Pool.ConfigurationFilename := MakeConfigurationFile(LFilename);
  ProgressBar.Visible := True;
  ToogleControlPanelEnabled(ProgressBar);
end;

procedure TFormBackground.ButtonNewParticipantClick(Sender: TObject);
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
  end;
end;

procedure TFormBackground.ButtonLoadConfigurationFileClick(Sender: TObject);
begin
  SetupFolders;         // todo: pass filename id of loaded file into session.counters.loadfromfile
  OpenDialog1.InitialDir := Pool.BaseFileName;
  if OpenDialog1.Execute then begin
    Pool.ConfigurationFilename := LoadConfigurationFile(OpenDialog1.FileName);
    ProgressBar.Max := 1;
    ProgressBar.StepIt;
    ProgressBar.Visible := True;
  end;
end;

procedure TFormBackground.ButtonMiscClick(Sender: TObject);
begin
  FormMisc.ShowModal;
end;

procedure TFormBackground.BeginSession(Sender: TObject);
begin
  if Assigned(EyeTracker) then begin
    EyeTracker.StartRecording;
  end;
  TLogger.SetHeader(SessionName, ParticipantFolderName);
end;

procedure TFormBackground.ComboBoxDesignFolderEditingDone(Sender: TObject);
begin
  with Pool, ComboBoxDesignFolder do begin
    DesignBasePath := Items[ItemIndex];
  end;
  ComboBoxCondition.Clear;
  GetDesignFilesFor(ComboBoxCondition.Items);
end;

procedure TFormBackground.EndSession(Sender: TObject);
begin

end;

procedure TFormBackground.CloseSDLApp(Sender: TObject);
begin
  if Assigned(EyeTracker) then begin
    EyeTracker.StopRecording;
  end;
  TLogger.SetFooter;
  SDLSession.Free;
  SDLApp.Free;
  FreeConfigurationFile;
  ToogleControlPanelEnabled;
  ProgressBar.Visible := False;
  Controllers.Free;
end;

procedure TFormBackground.IniPropStorage1RestoreProperties(Sender: TObject);
begin
  SetupFolders;
end;

procedure TFormBackground.IniPropStorage1StoredValues0Restore(
  Sender: TStoredValue; var Value: TStoredType);
begin
  GetDesignFoldersFor(ComboBoxDesignFolder.Items);
  with Pool, ComboBoxDesignFolder do begin
    DesignBasePath := Items[Value.ToInteger];
  end;
end;

procedure TFormBackground.IniPropStorage1StoredValues0Save(
  Sender: TStoredValue; var Value: TStoredType);
begin
  Value := ComboBoxDesignFolder.ItemIndex.ToString;
end;

procedure TFormBackground.IniPropStorage1StoredValues1Restore(
  Sender: TStoredValue; var Value: TStoredType);
begin
  GetDesignFilesFor(ComboBoxCondition.Items);
  ComboBoxCondition.ItemIndex := Value.ToInteger;
end;

procedure TFormBackground.IniPropStorage1StoredValues1Save(
  Sender: TStoredValue; var Value: TStoredType);
begin
  Value := ComboBoxCondition.ItemIndex.ToString;
end;

procedure TFormBackground.MenuItemCopyPNGFilesClick(Sender: TObject);
begin
  E1CopyRandomImagesToParticipantFolder;
end;

procedure TFormBackground.MenuItemRemoveParticipantClick(Sender: TObject);
begin
  with ComboBoxParticipant do
    Items.Delete(ItemIndex);
end;

procedure TFormBackground.AssignGlobalVariables;
begin
  TestMode := FormMisc.CheckBoxTestMode.Checked;

  GlobalTrialParameters.Cursor := 1;
  GlobalTrialParameters.FixedComparisonPosition := 7;

  with GlobalTrialParameters,
       FormMisc.CheckBoxShowModalFormForSpeechResponses do
    ShowModalFormForSpeechResponses := Checked;

  with GlobalTrialParameters, FormMisc.ComboBoxShouldRestartAt do
    ShouldRestartAtBlockStart := ItemIndex = 0;

  with GlobalTrialParameters, FormMisc.ComboBoxAudioPromptForText do
    AudioPromptForText := Items[ItemIndex];

  with GlobalTrialParameters, FormMisc.ComboBoxFontName do
    FontName := Items[ItemIndex];

  with GlobalTrialParameters, FormMisc.SpinEditFontSize do
    FontSize := Value;

  with GlobalTrialParameters, FormMisc.SpinEditRecordingSeconds do
    RecordingSeconds := Value;

  with GlobalTrialParameters, FormMisc.SpinEditInterTrialInterval do
    InterTrialInterval := Value.SecondsToMiliseconds;

  with GlobalTrialParameters, FormMisc.SpinEditLimitedHold do
    LimitedHold := Value.MinutesToMiliseconds;

  with GlobalTrialParameters, FormMisc.SpinEditTimeOut do
    TimeOutInterval := Value.SecondsToMiliseconds;

  with GlobalTrialParameters, FormMisc.ComboBoxAudioFolder do
    Pool.AudioBasePath := Items[ItemIndex];

  with GlobalTrialParameters, FormMisc.ComboBoxFixedSamplePosition do begin
    GridOrientation := goCustom;
    case ItemIndex of
      1: begin // centralize sample, use 4 corners for comparisions
        FixedSamplePosition := 4;
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
end;

procedure TFormBackground.ToogleControlPanelEnabled(AException: TComponent);
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

function TFormBackground.ParticipantFolderName: string;
begin
  Pool.Counters.Subject := ComboBoxParticipant.ItemIndex;
  Result := Pool.Counters.Subject.ToString +'-'+
      ComboBoxParticipant.Items[Pool.Counters.Subject] +
      DirectorySeparator;
end;

function TFormBackground.SessionName: string;
begin
  Result := 'Sessão';
end;

function TFormBackground.SetupFolders: Boolean;
begin
  Pool.ImageBasePath := ParticipantFolderName;
  Pool.BaseFileName :=
    ConcatPaths([Pool.DataRootBasePath, ParticipantFolderName]);
  Pool.DataResponsesBasePath :=
    ConcatPaths([Pool.BaseFileName, Pool.ResponsesBasePath]);

  Result :=
    ForceDirectories(Pool.BaseFileName) and
    ForceDirectories(Pool.DataResponsesBasePath);
end;

function TFormBackground.Validated: Boolean;
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

  if FormMisc.ComboBoxMonitor.ItemIndex = -1 then begin
    ShowMessage('Escolha um monitor.');
    Exit;
  end;

  if Pool.ConfigurationFilename.IsEmpty then begin
    ShowMessage('Crie uma nova sessão ou carregue uma sessão interrompida.');
    Exit;
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

end.

