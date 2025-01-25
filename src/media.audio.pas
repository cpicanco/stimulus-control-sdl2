{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit media.audio;

{$mode ObjFPC}{$H+}

interface

procedure ListAvailableVoices;
procedure Speak(const ASSML:string);
procedure SpeakToFile(const ASSML:string; AFilename:string);
function GetMediaPath(FileName: string): string;

implementation

uses
  Classes, SysUtils
  //, sdl.app.output
  , SpeechLib_TLB;

function GetMediaPath(FileName: string): string;
var
  MediaPath: string;
begin
  // Check if the environment variable for media path is set
  MediaPath := GetEnvironmentVariable('STIMULUS_CONTROL_MEDIA_PATH');

  if MediaPath = '' then begin
    // MEDIA_PATH environment variable is not set, use default path
    MediaPath := ConcatPaths([ExtractFilePath(ParamStr(0)), 'media']);
  end;

  // Ensure the media path exists
  if not DirectoryExists(MediaPath) then
  begin
    CreateDir(MediaPath);
  end;

  // Combine the media path with the filename
  MediaPath := ConcatPaths([MediaPath, FileName]);

  // Return the combined path
  Result := MediaPath;
end;


procedure ListAvailableVoices;
var
  LSpVoice: SpVoice;
  TokenEnum: ISpeechObjectTokens;
  Token: ISpeechObjectToken;
  AttrValue: WideString;
  i: Integer;
begin
  LSpVoice := CoSpVoice.Create;
  try
    TokenEnum := LSpVoice.GetVoices('', '');
    for i := 0 to TokenEnum.Count-1 do
    begin
      Token := TokenEnum.Item(i);
      AttrValue := Token.GetDescription(0);
      Print(AttrValue);
    end;
  finally
    LSpVoice := nil;
  end;
end;

procedure Speak(const ASSML:string);
var
  LSpVoice: SpVoice;
begin
  LSpVoice := CoSpVoice.Create;
  try
    LSpVoice.Speak(ASSML, SVSFDefault);
  finally
    LSpVoice := nil;
  end;
end;

procedure SpeakToFile(const ASSML:string; AFilename:string);
var
  LSpFileStream: SpFileStream;
  LSpVoice: SpVoice;
begin
  LSpFileStream := CoSpFileStream.Create;
  LSpVoice := CoSpVoice.Create;
  try
    LSpFileStream.Format.Type_ := SAFT44kHz16BitStereo;
    LSpFileStream.Open(AFileName, SSFMCreateForWrite, False);
    LSpVoice.AudioOutputStream := LSpFileStream;
    LSpVoice.Speak(ASSML, SVSFDefault);
    LSpFileStream.Close;
  finally
    LSpVoice := nil;
    LSpFileStream := nil;
  end;
end;

end.

