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
  IniPropStorage;

type

  { TFormBackground }

  TFormBackground = class(TForm)
    ButtonNewParticipant: TButton;
    ButtonLoadConfigurationFile: TButton;
    ButtonNewConfigurationFile: TButton;
    ButtonRunSession: TButton;
    ComboBoxCondition: TComboBox;
    ComboBoxParticipant: TComboBox;
    IniPropStorage1: TIniPropStorage;
    OpenDialog1: TOpenDialog;
    Panel1: TPanel;
    procedure ButtonLoadConfigurationFileClick(Sender: TObject);
    procedure ButtonNewConfigurationFileClick(Sender: TObject);
    procedure ButtonNewParticipantClick(Sender: TObject);
    procedure ButtonRunSessionClick(Sender: TObject);
    procedure BeginSession(Sender: TObject);
    procedure EndSession(Sender : TObject);
    procedure CloseSDLApp(Sender : TObject);
    procedure FormCreate(Sender: TObject);
  private
    //FEyeLink : TEyeLink;
    function ParticipantFolderName : string;
    function SessionName : string;
    function Validated : Boolean;
  public

  end;

var
  FormBackground: TFormBackground;

implementation

{$R *.lfm}

uses
  FileUtil
  , StrUtils
  , session
  , session.pool
  , session.loggers
  , session.fileutils
  , experiments
  , sdl.app
  ;

{ TFormBackground }

procedure TFormBackground.ButtonRunSessionClick(Sender: TObject);
begin
  if not Validated then Exit;

  SDLApp := TSDLApplication.Create(@Pool.AppName[1], 0);
  SDLApp.SetupEvents;
  SDLApp.SetupAudio;
  SDLApp.SetupText;
  SDLApp.OnClose := @CloseSDLApp;

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
  if ComboBoxCondition.Items.Count = 0 then begin
    ShowMessage('A pasta de condições (design) está vazia.');
    Exit;
  end;
  if ComboBoxCondition.ItemIndex = -1 then begin
    ShowMessage('Escolha uma condição.');
    Exit;
  end else begin
    with ComboBoxCondition do begin
      LFilename := Items[ItemIndex];
    end;
  end;
  ConfigurationFilename := Experiments.MakeConfigurationFile(LFilename);
end;

procedure TFormBackground.ButtonNewParticipantClick(Sender: TObject);
var
  LNewParticipant : string;
begin
  with ComboBoxParticipant do begin
    LNewParticipant := InputBox(Pool.AppName,
                   'Nome: mínimo de 3 caracteres',
                   '');
    if LNewParticipant.IsEmpty or (Length(LNewParticipant) < 3) then Exit;

    Items.Append(LNewParticipant);
  end;
end;

procedure TFormBackground.ButtonLoadConfigurationFileClick(Sender: TObject);
begin
  if OpenDialog1.Execute then
    ConfigurationFilename := LoadConfigurationFile(OpenDialog1.FileName);
end;

procedure TFormBackground.BeginSession(Sender: TObject);
begin
  TLogger.SetHeader(SessionName, ParticipantFolderName);
  CopyFile(ConfigurationFilename, Pool.BaseFilename+'.ini');
end;

procedure TFormBackground.EndSession(Sender: TObject);
begin

end;

procedure TFormBackground.CloseSDLApp(Sender: TObject);
begin
  TLogger.SetFooter;
  SDLSession.Free;
  SDLApp.Free;
end;

procedure TFormBackground.FormCreate(Sender: TObject);
begin
  GetDesignFilesFor(ComboBoxCondition.Items);
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

function TFormBackground.Validated: Boolean;
  function SetupFolders : Boolean;
  begin
    Pool.BaseFileName := Pool.RootData +
      ParticipantFolderName;

    Pool.RootDataResponses := Pool.RootData +
      ParticipantFolderName + Pool.ResponsesBasePath;

    Result :=
      ForceDirectories(Pool.BaseFileName) and
      ForceDirectories(Pool.RootDataResponses);
  end;

  function SetupParticipantID : Boolean;
  var
    LParticipantID: TStringList;
    LIDFile : string;
    LID : string;
  begin
    LIDFile := Pool.RootData + ParticipantFolderName + 'ID';
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

  if ConfigurationFilename.IsEmpty then begin
    ShowMessage('Crie uma nova sessão ou abra uma pronta.');
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

